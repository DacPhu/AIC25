import axios from "axios";

const PORT = import.meta.env.VITE_PORT || 5000;

export async function search(
  q,
  offset,
  limit,
  nprobe,
  model,
  temporal_k,
  ocr_weight,
  ocr_threshold,
  max_interval,
  selected,
) {
  const res = await axios.get(`http://127.0.0.1:${PORT}/api/search`, {
    params: {
      q: q,
      offset: offset,
      limit: limit,
      nprobe: nprobe,
      model: model,
      temporal_k: temporal_k,
      ocr_weight: ocr_weight,
      ocr_threshold: ocr_threshold,
      max_interval: max_interval,
      selected: selected,
    },
  });
  return res.data;
}
export async function searchSimilar(
  id,
  offset,
  limit,
  nprobe,
  model,
  temporal_k,
  ocr_weight,
  ocr_threshold,
  max_interval,
) {
  const res = await axios.get(`http://127.0.0.1:${PORT}/api/similar`, {
    params: {
      id: id,
      offset: offset,
      limit: limit,
      nprobe: nprobe,
      model: model,
      temporal_k: temporal_k,
      ocr_weight: ocr_weight,
      ocr_threshold: ocr_threshold,
      max_interval: max_interval,
    },
  });
  return res.data;
}

export async function getFrameInfo(videoId, frameId) {
  const res = await axios.get(`http://127.0.0.1:${PORT}/api/frame_info`, {
    params: {
      video_id: videoId,
      frame_id: frameId,
    },
  });
  return res.data;
}

export async function getAvailableModels() {
  const res = await axios.get(`http://127.0.0.1:${PORT}/api/models`);
  return res.data;
}
