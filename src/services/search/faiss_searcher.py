import hashlib
import logging
import re
import time
import os
from copy import deepcopy

import librosa
from thefuzz import fuzz

from config import GlobalConfig
from services.analyse.features import CLIP, AudioExtractor
from services.analyse.features.sentence_transformer import SentenceTransformerExtractor
from utils import create_filter_func

from ..index import FAISSDatabase
from .base import BaseSearch


class Searcher(BaseSearch):
    cache = {}

    def _safe_get_attribute(self, obj, attr, default=None):
        """Safely get attribute from object or dictionary"""
        if hasattr(obj, attr):
            return getattr(obj, attr)
        elif hasattr(obj, "get"):
            return obj.get(attr, default)
        else:
            return default

    def __init__(self, collection_name):
        self._logger = logging.getLogger("searcher")

        work_dir = os.getenv("AIC25_WORK_DIR", ".")
        self._database = FAISSDatabase(collection_name, db_dir=work_dir)
        self._models = {}

        for model in GlobalConfig.get("webui", "features") or []:
            model_name = model["name"].lower()
            if model_name == "clip":
                pretrained_model = model["pretrained_model"]
                self._models[model_name] = CLIP(pretrained_model)
            elif model_name == "audio":
                pretrained_model = model.get(
                    "pretrained_model", "facebook/wav2vec2-base-960h"
                )
                self._models[model_name] = AudioExtractor(pretrained_model)
            elif (
                model_name == "sentence_transformer"
                or model_name == "sentence-transformer"
            ):
                pretrained_model = model.get(
                    "pretrained_model", "all-MiniLM-L6-v2"  # Default lightweight model
                )
                device = model.get("device", "auto")
                self._models[model_name] = SentenceTransformerExtractor(
                    pretrained_model, device
                )

        if len(self._models) == 0:
            self._logger.error(
                f'No models found in "{GlobalConfig.CONFIG_FILE}". Check your "{GlobalConfig.CONFIG_FILE}"'
            )

    def get(self, param_id):
        result = self._database.get(param_id)
        return [result] if result else []

    def get_models(self):
        return list(self._models.keys())

    def _process_query(self, query):
        video_match = re.search('video:((".+?")|\\S+)\\s?', query)
        video_ids = (
            video_match.group().replace("video:", "", 1).strip('" ').split(",")
            if video_match is not None
            else []
        )
        if video_match is not None:
            query = query.replace(video_match.group(), "", 1)

        queries = query.split(";")
        processed = {
            "queries": [],
            "advance": [],
            "video_ids": video_ids,
        }
        for q in queries:
            q = q.strip()
            ocr = []
            while True:
                match = re.search('OCR:((".+?")|\\S+)\\s?', q)
                if match is None:
                    break
                ocr.append(match.group().replace("OCR:", "", 1).strip('" ').lower())
                q = q.replace(match.group(), "")

            processed["queries"].append(q)
            processed["advance"].append({})

            if len(ocr) > 0:
                processed["advance"][-1]["ocr"] = ocr
        return processed

    def _process_advance(self, advance_query, result, ocr_weight, ocr_threshold):
        if "ocr" not in advance_query:
            return result
        query_ocr = advance_query["ocr"]
        res = deepcopy(result)

        for i, record in enumerate(result):
            ocr_distance = 0
            for query_text in query_ocr:
                # Handle both SearchResult objects and dictionary results
                if hasattr(record, "entity"):
                    # SearchResult object
                    entity = record.entity
                    if isinstance(entity, dict):
                        ocr = entity.get("ocr", [])
                    else:
                        # If entity is not a dict, try to get the ocr attribute
                        ocr = getattr(entity, "ocr", [])
                else:
                    # Dictionary result
                    ocr = record.get("entity", {}).get("ocr", [])
                ocr_text_distance = 0
                cnt = 0
                for text in ocr:
                    text_content = (
                        text[-2]
                        if isinstance(text, list) and len(text) > 2
                        else str(text)
                    )
                    partial_ratio = fuzz.partial_ratio(
                        query_text.lower(), text_content.lower()
                    )
                    if partial_ratio > ocr_threshold:
                        cnt += 1
                        ocr_text_distance += partial_ratio / 100

                if cnt > 0:
                    ocr_text_distance /= cnt
                ocr_distance += ocr_text_distance

            if len(query_ocr) > 0:
                ocr_distance /= len(query_ocr)

            if hasattr(res[i], "distance"):
                res[i].distance = (record.distance + ocr_distance * ocr_weight) / (
                    1 + ocr_weight
                )
            else:
                if hasattr(res[i], "get"):
                    res[i]["distance"] = (
                        record["distance"] + ocr_distance * ocr_weight
                    ) / (1 + ocr_weight)
                else:
                    new_distance = (record.distance + ocr_distance * ocr_weight) / (
                        1 + ocr_weight
                    )
                    setattr(res[i], "distance", new_distance)

        res = sorted(
            res,
            key=lambda x: self._safe_get_attribute(x, "distance", 0),
            reverse=True,
        )
        return res

    def _combine_temporal_results(self, results, temporal_k, max_interval):
        best = None
        for i in range(len(results)):
            res = results[i]
            for j in range(len(res)):
                if hasattr(res[j], "frame_id"):
                    frame_id = res[j].frame_id
                elif hasattr(res[j], "entity"):
                    entity = res[j].entity
                    if isinstance(entity, dict):
                        frame_id = entity.get("frame_id", "")
                    else:
                        frame_id = getattr(entity, "frame_id", "")
                else:
                    frame_id = res[j].get("entity", {}).get("frame_id", "")
                video_id, frame_id_num = frame_id.split("#")
                video_id = video_id.replace("L", "").replace("_V", "")
                video_id = int(video_id)
                frame_id_num = int(frame_id_num)

                if hasattr(results[i][j], "_id"):
                    results[i][j]._id = (video_id, frame_id_num)
                else:
                    # For dictionary results, use item assignment
                    if hasattr(results[i][j], "get"):
                        results[i][j]["_id"] = (video_id, frame_id_num)
                    else:
                        # For objects without _id attribute, add it dynamically
                        setattr(results[i][j], "_id", (video_id, frame_id_num))

        for res in results[::-1]:
            if best is None:
                best = res[:temporal_k]
                continue
            tmp = []
            res = sorted(res, key=lambda x: self._safe_get_attribute(x, "_id"))
            best = sorted(best, key=lambda x: self._safe_get_attribute(x, "_id"))
            l = 0
            r = 0
            for cur in res:
                cur_id = self._safe_get_attribute(cur, "_id")
                cur_vid, cur_fid = cur_id
                cur_fid = int(cur_fid)

                while l < len(best):
                    next_id = self._safe_get_attribute(best[l], "_id")
                    next_vid, next_fid = next_id
                    next_fid = int(next_fid)
                    if next_vid > cur_vid or (
                        next_vid == cur_vid and next_fid > cur_fid
                    ):
                        break
                    else:
                        l += 1

                while r < len(best):
                    next_id = self._safe_get_attribute(best[r], "_id")
                    next_vid, next_fid = next_id
                    next_fid = int(next_fid)
                    if next_vid > cur_vid or (
                        next_vid == cur_vid and next_fid > cur_fid + max_interval
                    ):
                        break
                    else:
                        r += 1

                for i in range(l, r):
                    cur_distance = self._safe_get_attribute(cur, "distance", 0)
                    best_distance = self._safe_get_attribute(best[i], "distance", 0)

                    new_item = deepcopy(cur)
                    if hasattr(new_item, "distance"):
                        new_item.distance = cur_distance + best_distance
                    else:
                        if hasattr(new_item, "get"):
                            new_item["distance"] = cur_distance + best_distance
                        else:
                            setattr(new_item, "distance", cur_distance + best_distance)
                    tmp.append(new_item)

            highest = {}
            for cur in tmp:
                cur_id = self._safe_get_attribute(cur, "_id")
                cur_distance = self._safe_get_attribute(cur, "distance", 0)
                if cur_id not in highest or cur_distance > self._safe_get_attribute(
                    highest[cur_id], "distance", 0
                ):
                    highest[cur_id] = cur
            tmp = list(highest.values())
            tmp = sorted(
                tmp,
                key=lambda x: self._safe_get_attribute(x, "distance", 0),
                reverse=True,
            )
            best = tmp[:temporal_k]

        return best

    def _simple_search(self, processed, filter_expr, offset, limit, nprobe, model):
        import torch

        text_features_tensor = self._models[model].get_text_features(
            processed["queries"]
        )

        # Handle tensor conversion properly
        if torch.is_tensor(text_features_tensor):
            text_features = text_features_tensor.cpu().detach().numpy().tolist()
        else:
            text_features = text_features_tensor.tolist()
        filter_func = self._combine_videos_filter(filter_expr, processed["video_ids"])

        results = self._database.search(
            text_features,
            filter_func,
            offset,
            limit,
            nprobe,
        )[0]

        results_dict = []
        for result in results:
            results_dict.append({"distance": result.distance, "entity": result.entity})

        res = {
            "results": results_dict,
            "total": self._database.get_total(),
            "offset": offset,
        }
        return res

    def _combine_videos_filter(self, filter_expr, video_ids):
        """Create filter function for video IDs"""

        def filter_func(entity):
            if filter_expr:
                original_filter = create_filter_func(filter_expr)
                if original_filter and not original_filter(entity):
                    return False

            if video_ids:
                # Handle both dict and object entities
                if isinstance(entity, dict):
                    frame_id = entity.get("frame_id", "")
                else:
                    frame_id = getattr(entity, "frame_id", "")

                for video_id in video_ids:
                    if frame_id.startswith(f"{video_id.strip()}#"):
                        return True
                return False

            return True

        return filter_func if (filter_expr or video_ids) else None

    def _complex_search(
        self,
        processed,
        filter_expr,
        offset,
        limit,
        nprobe,
        model,
        temporal_k,
        ocr_weight,
        ocr_threshold,
        max_interval,
    ):
        params = {
            "filter": filter_expr,
            "nprobe": nprobe,
            "model": model,
            "temporal_k": temporal_k,
            "ocr_weight": ocr_weight,
            "ocr_threshold": ocr_threshold,
            "max_interval": max_interval,
        }
        self._logger.debug(processed)
        self._logger.debug(params)

        query_hash = hashlib.sha256(
            (f"complex:{repr(processed)}{repr(params)}").encode("utf-8")
        ).hexdigest()

        if query_hash in self.cache:
            combined_results = self.cache[query_hash]
        else:
            import torch

            text_features_tensor = self._models[model].get_text_features(
                processed["queries"]
            )

            # Handle tensor conversion properly
            if torch.is_tensor(text_features_tensor):
                text_features = text_features_tensor.cpu().detach().numpy().tolist()
            else:
                text_features = text_features_tensor.tolist()
            filter_func = self._combine_videos_filter(
                filter_expr, processed["video_ids"]
            )

            st = time.time()
            results = self._database.search(
                text_features,
                filter_func,
                0,
                temporal_k,
                nprobe,
            )
            en = time.time()
            self._logger.debug(f"{en - st:.4f} seconds to search results")

            for i in range(len(processed["queries"])):
                results[i] = self._process_advance(
                    processed["advance"][i],
                    results[i],
                    ocr_weight,
                    ocr_threshold,
                )

            st = time.time()
            combined_results = self._combine_temporal_results(
                results, temporal_k, max_interval
            )
            en = time.time()
            self._logger.debug(f"{en - st:.4f} seconds to combine results")
            self.cache[query_hash] = combined_results

        if combined_results is not None and offset < len(combined_results):
            results = combined_results[offset : offset + limit]
        else:
            results = []

        results_dict = []
        for result in results:
            if hasattr(result, "distance"):
                results_dict.append(
                    {"distance": result.distance, "entity": result.entity}
                )
            else:
                results_dict.append(result)

        res = {
            "results": results_dict,
            "total": len(combined_results or []),
            "offset": offset,
        }
        return res

    def _get_videos(self, video_ids, offset, limit, selected):
        query_hash = hashlib.sha256(
            f"video:{repr(video_ids)}".encode("utf-8")
        ).hexdigest()

        if query_hash in self.cache:
            videos = self.cache[query_hash]
        elif len(video_ids) == 0:
            videos = []
        else:
            filter_func = self._combine_videos_filter("", video_ids)
            videos_data = self._database.query(filter_func, 0, 10000)
            videos = [{"entity": x} for x in videos_data]
            videos = sorted(videos, key=lambda x: x["entity"]["frame_id"])
            self.cache[query_hash] = videos

        if selected:
            for i, video in enumerate(videos):
                if selected == video["entity"]["frame_id"]:
                    offset = (i // limit) * limit
                    break

        res = {
            "results": videos[offset : offset + limit],
            "total": len(videos),
            "offset": offset,
        }
        return res

    def search(
        self,
        q: str,
        param_filter: str = "",
        offset: int = 0,
        limit: int = 50,
        nprobe: int = 8,
        model: str = "clip",
        temporal_k: int = 10000,
        ocr_weight: float = 1.0,
        ocr_threshold: int = 40,
        max_interval: int = 250,
        selected: str | None = None,
    ):
        processed = self._process_query(q)
        no_query = all([len(x) == 0 for x in processed["queries"]])
        no_advance = all([len(x) == 0 for x in processed["advance"]])

        if no_query and no_advance:
            self._logger.debug(f"Get videos: {q}")
            return self._get_videos(processed["video_ids"], offset, limit, selected)
        elif len(processed["queries"]) == 1 and no_advance:
            self._logger.debug(f"Simple search: {q}")
            return self._simple_search(
                processed, param_filter, offset, limit, nprobe, model
            )
        else:
            self._logger.debug(f"Complex search: {q}")
            return self._complex_search(
                processed,
                param_filter,
                offset,
                limit,
                nprobe,
                model,
                temporal_k,
                ocr_weight,
                ocr_threshold,
                max_interval,
            )

    def search_similar(
        self,
        param_id: str,
        offset: int = 0,
        limit: int = 50,
        nprobe: int = 8,
        model: str = "clip",
    ):
        record = self._database.get(param_id)
        if not record:
            return {"results": [], "total": 0, "offset": 0}

        image_features = [record[model]]

        results = self._database.search(image_features, None, offset, limit, nprobe)[0]

        results_dict = []
        for result in results:
            results_dict.append({"distance": result.distance, "entity": result.entity})

        res = {
            "results": results_dict,
            "total": self._database.get_total(),
            "offset": offset,
        }
        return res

    def search_by_audio(
        self,
        audio_file_path: str,
        offset: int = 0,
        limit: int = 50,
        nprobe: int = 8,
        model: str = "audio",
    ):
        """
        Search for similar audio segments given an audio file.

        Args:
            audio_file_path: Path to the audio file to search with
            offset: Starting offset for results
            limit: Maximum number of results
            nprobe: FAISS search parameter
            model: Audio model to use (default: "audio")

        Returns:
            Search results similar to regular search
        """
        if model not in self._models:
            self._logger.error(f"Audio model '{model}' not available")
            return {"results": [], "total": 0, "offset": 0}

        try:
            audio_segment, sr = librosa.load(
                audio_file_path, sr=16000, duration=4.0, mono=True
            )

            # Extract Wav2Vec2 features using the audio model
            query_features = self._models[model]._extract_wav2vec2_features(
                audio_segment
            )

            query_vector = [query_features.tolist()]

            # Perform vector search using audio features
            results = self._database.search(
                query_vector,
                filter_func=None,
                offset=offset,
                limit=limit,
                nprobe=nprobe,
            )

            if results and len(results) > 0:
                results_dict = []
                for result in results[0]:  # First query results
                    results_dict.append(
                        {"distance": result.distance, "entity": result.entity}
                    )

                return {
                    "results": results_dict,
                    "total": self._database.get_total(),
                    "offset": offset,
                    "search_type": "audio",
                }

            return {"results": [], "total": 0, "offset": offset, "search_type": "audio"}

        except Exception as e:
            self._logger.error(f"Audio search failed: {e}")
            return {"results": [], "total": 0, "offset": offset, "error": str(e)}
