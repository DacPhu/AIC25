import {
  useLoaderData,
  Form,
  useSubmit,
  useOutletContext,
  useNavigation,
} from "react-router-dom";
import classNames from "classnames";
import { useEffect, useState, useMemo, useCallback } from "react";

import { search } from "@/services/search";
import { FrameItem, FrameContainer } from "@/components/Frame";
import VirtualGrid from "@/components/VirtualGrid";
import AudioSearch from "@/components/AudioSearch";
import { usePlayVideo } from "@/components/VideoPlayer";
import { useCache } from "@/hooks/useCache";
import { Dropdown, Editable, Checkbox } from "@/components/Filter";
import PreviousButton from "@/assets/previous-btn.svg";
import NextButton from "@/assets/next-btn.svg";
import HomeButton from "@/assets/home-btn.svg";
import SpinIcon from "@/assets/spin.svg";

import {
  nlist,
  limitOptions,
  nprobeOption,
  temporal_k_default,
  ocr_weight_default,
  ocr_threshold_default,
  max_interval_default,
  use_sentence_transformer_default,
  sentence_transformer_options,
} from "@/resources/options";

export async function loader({ request }) {
  const url = new URL(request.url);
  const searchParams = url.searchParams;

  const q = searchParams.get("q");
  const _offset = searchParams.get("offset") || 0;
  const selected = searchParams.get("selected") || undefined;
  const limit = searchParams.get("limit") || limitOptions[0];
  const nprobe = searchParams.get("nprobe") || nprobeOption[0];
  const model = searchParams.get("model") || undefined;
  const temporal_k = searchParams.get("temporal_k") || temporal_k_default;
  const ocr_weight = searchParams.get("ocr_weight") || ocr_weight_default;
  const ocr_threshold =
    searchParams.get("ocr_threshold") || ocr_threshold_default;
  const max_interval = searchParams.get("max_interval") || max_interval_default;

  const { total, frames, params, offset } = await search(
    q,
    _offset,
    limit,
    nprobe,
    model,
    temporal_k,
    ocr_weight,
    ocr_threshold,
    max_interval,
    selected,
  );
  const query = q ? { q } : {};

  return {
    query,
    params,
    selected,
    offset,
    data: { total, frames },
  };
}

export default function Search() {
  const navigation = useNavigation();
  const { modelOptions } = useOutletContext();
  const submit = useSubmit();
  const { query, params, offset, data, selected } = useLoaderData();
  const playVideo = usePlayVideo();
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [useVirtualScroll, setUseVirtualScroll] = useState(true);
  const [activeSearchTab, setActiveSearchTab] = useState<'text' | 'audio'>('text');

  const { q = "", id = null } = query;
  const { limit, nprobe, model } = params;

  const { total, frames } = data;
  const empty = frames.length === 0;
  
  // Memoize expensive computations
  const isLoading = navigation.state === "loading";
  const currentPage = Math.floor(offset / limit) + 1;
  const hasNextPage = total > offset + frames.length;

  useEffect(() => {
    // Set correct values
    const searchBar = document.querySelector("#search-bar");
    if (searchBar) {
      searchBar.value = q || "";
      searchBar.focus();
    }

    document.title = q ? `${q} - Page ${currentPage}` : 'Search';
  }, [q, currentPage]);

  for (const [k, v] of Object.entries(params)) {
    useEffect(() => {
      document.querySelector(`#${k}`).value = v;
    }, [v]);
  }

  // Add hotkeys
  useEffect(() => {
    const handleKeyDown = (e) => {
      switch (e.keyCode) {
        case 191:
          const filterBar = document.querySelector("#search-area");
          filterBar.scrollIntoView();
          const searchBar = document.querySelector("#search-bar");
          if (searchBar !== document.activeElement) {
            e.preventDefault();
            searchBar.focus();
            return false;
          }
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, []);
  // Memoized event handlers
  const handleKeyDown = useCallback((e) => {
    switch (e.keyCode) {
      case 38:
        e.preventDefault();
        goToPreviousPage();
        return;
      case 40:
        e.preventDefault();
        goToNextPage();
        return;
    }
  }, [offset, limit]);
  
  useEffect(() => {
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [handleKeyDown]);

  const goToFirstPage = () => {
    submit({
      ...query,
      ...params,
      offset: 0,
    });
  };
  const goToPreviousPage = () => {
    submit({
      ...query,
      ...params,
      offset: Math.max(parseInt(offset) - parseInt(limit), 0),
    });
  };
  const goToNextPage = () => {
    if (!empty) {
      submit({
        ...query,
        ...params,
        offset: parseInt(offset) + parseInt(limit),
      });
    }
  };

  // Memoized handlers to prevent unnecessary re-renders
  const handleOnPlay = useCallback((frame) => {
    playVideo(frame);
  }, [playVideo]);
  
  const handleOnSearchSimilar = useCallback((frame) => {
    submit({ id: frame.id, ...params }, { action: "/similar" });
  }, [submit, params]);

  const handleOnSelect = useCallback((frame) => {
    setSelectedFrame(frame.id);
    submit(
      {
        q: "video:" + frame.video_id,
        ...params,
        selected: frame.id,
      },
      { action: "/search" },
    );
  }, [submit, params, setSelectedFrame]);
  const handleOnChangeParams = (e) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const data = {};
    for (const [k, v] of formData.entries()) {
      if (k === "use_sentence_transformer") {
        data[k] = v === "on" || v === "true" || v === true;
      } else {
        data[k] = v;
      }
    }
    
    // Ensure checkbox is included even when unchecked (FormData omits unchecked checkboxes)
    if (!formData.has("use_sentence_transformer")) {
      data["use_sentence_transformer"] = false;
    }
    if (selected) {
      submit({
        ...query,
        ...data,
        selected,
      });
    } else {
      submit({
        ...query,
        ...data,
      });
    }
  };
  const handleOnSearch = (e) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const data = {};
    for (const [k, v] of formData.entries()) {
      // Convert checkbox values to proper booleans
      if (k === "use_sentence_transformer") {
        data[k] = v === "on" || v === "true" || v === true;
      } else {
        data[k] = v;
      }
    }
    
    // Ensure checkbox is included even when unchecked (FormData omits unchecked checkboxes)
    if (!formData.has("use_sentence_transformer")) {
      data["use_sentence_transformer"] = false;
    }
    
    document.activeElement.blur();
    submit(
      {
        ...data,
        ...params,
      },
      { action: "/search" },
    );
  };
  // Memoized render item function for virtual grid
  const renderFrameItem = useCallback((frame, index) => (
    <FrameItem
      key={frame.id}
      video_id={frame.video_id}
      frame_id={frame.frame_id}
      thumbnail={frame.frame_uri}
      onPlay={() => handleOnPlay(frame)}
      onSearchSimilar={() => handleOnSearchSimilar(frame)}
      onSelect={() => handleOnSelect(frame)}
      selected={selectedFrame === frame.id}
    />
  ), [handleOnPlay, handleOnSearchSimilar, handleOnSelect, selectedFrame]);
  
  const loadMoreData = useCallback(() => {
    if (hasNextPage && !isLoading) {
      goToNextPage();
    }
  }, [hasNextPage, isLoading]);

  return (
    <div id="search-area" className="flex flex-col w-full h-screen">
      {/* Navigation Tabs */}
      <nav className="bg-white border-b border-gray-200">
        <div className="px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            <button
              onClick={() => setActiveSearchTab('text')}
              className={classNames(
                "py-4 px-1 border-b-2 font-medium text-sm",
                activeSearchTab === 'text'
                  ? "border-blue-500 text-blue-600"
                  : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
              )}
            >
              <div className="flex items-center space-x-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <span>Text Search</span>
              </div>
            </button>
            <button
              onClick={() => setActiveSearchTab('audio')}
              className={classNames(
                "py-4 px-1 border-b-2 font-medium text-sm",
                activeSearchTab === 'audio'
                  ? "border-blue-500 text-blue-600"
                  : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
              )}
            >
              <div className="flex items-center space-x-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M9 12a3 3 0 106 0 3 3 0 00-6 0z" />
                </svg>
                <span>Audio Search</span>
              </div>
            </button>
          </div>
        </div>
      </nav>

      {/* Search Parameters (always visible) */}
      <Form className="flex flex-col border-b border-gray-200 bg-gray-50" onSubmit={handleOnChangeParams}>
        <div className="py-2 px-5 self-stretch text-md justify-start items-center flex flex-row flex-wrap">
          <Dropdown name="nprobe" options={nprobeOption} />
          <Dropdown name="limit" options={limitOptions} />
          <Dropdown name="model" options={modelOptions} />
          <Checkbox name="use_sentence_transformer" defaultValue={use_sentence_transformer_default} label="Semantic Search" />
          <Editable name="temporal_k" defaultValue={temporal_k_default} />
          <Editable name="ocr_weight" defaultValue={ocr_weight_default} />
          <Editable name="ocr_threshold" defaultValue={ocr_threshold_default} />
          <Editable name="max_interval" defaultValue={max_interval_default} />
        </div>

        <input
          className="self-center h-fit text-md px-4 py-2 mb-2 border-2 border-gray-300 rounded-xl bg-white text-gray-800 hover:bg-gray-50 active:bg-gray-100"
          type="submit"
          value="Apply Parameters"
        />
      </Form>

      {/* Text Search Section */}
      {activeSearchTab === 'text' && (
        <div className="flex flex-col flex-1">
          <Form id="search-form" onSubmit={handleOnSearch}>
            <div className="flex flex-col p-4 space-y-3 bg-gray-100 border-b border-gray-200">
              <div className="flex flex-row space-x-5">
                <img
                  className={classNames("h-8 w-8 self-center", {
                    "visible animate-spin": navigation.state === "loading",
                    invisible: navigation.state !== "loading",
                  })}
                  src={SpinIcon}
                />
                <textarea
                  form="search-form"
                  autoComplete="off"
                  className="flex-grow text-lg p-3 border-2 rounded-2xl border-gray-300 bg-white text-gray-800 focus:border-blue-500 focus:bg-white focus:text-black focus:outline-none shadow-sm transition-all duration-200 focus:shadow-md"
                  name="q"
                  id="search-bar"
                  type="search"
                  placeholder="Search for text, objects, scenes, OCR content..."
                  onKeyDown={(e) => {
                    if (e.keyCode === 13 && e.shiftKey === false) {
                      e.preventDefault();
                      document.querySelector("#search-form input").click();
                    }
                  }}
                />
                <input
                  className="self-center text-lg py-2 px-6 border-2 border-gray-300 rounded-2xl bg-blue-600 text-white hover:bg-blue-700 active:bg-blue-800 shadow-md hover:shadow-lg transition-all duration-200 font-medium"
                  type="submit"
                  value="Search"
                />
              </div>
              
              {/* OCR Search Helper Section */}
              <div className="bg-white rounded-2xl p-4 border border-gray-200 shadow-sm">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-gray-700 flex items-center">
                    <svg className="w-4 h-4 mr-2 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    OCR Text Search
                  </h3>
                  <button
                    type="button"
                    className="text-xs text-blue-600 hover:text-blue-800"
                    onClick={() => {
                      const searchBar = document.querySelector("#search-bar");
                      const currentValue = searchBar.value;
                      const newValue = currentValue ? `${currentValue} OCR:""` : 'OCR:""';
                      searchBar.value = newValue;
                      // Position cursor between quotes
                      const cursorPos = newValue.lastIndexOf('"');
                      searchBar.setSelectionRange(cursorPos, cursorPos);
                      searchBar.focus();
                    }}
                  >
                    Add OCR Query
                  </button>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-gray-600">
                  <div>
                    <span className="font-medium">Examples:</span>
                    <ul className="mt-1 space-y-1">
                      <li className="flex items-center">
                        <code className="bg-gray-100 px-1 rounded mr-2">OCR:"hello"</code>
                        <span>Find text "hello"</span>
                      </li>
                      <li className="flex items-center">
                        <code className="bg-gray-100 px-1 rounded mr-2">OCR:"price"</code>
                        <span>Find price mentions</span>
                      </li>
                    </ul>
                  </div>
                  <div>
                    <span className="font-medium">Mixed search:</span>
                    <ul className="mt-1 space-y-1">
                      <li className="flex items-center">
                        <code className="bg-gray-100 px-1 rounded mr-2 text-xs">person OCR:"name"</code>
                        <span>Visual + text</span>
                      </li>
                      <li className="flex items-center">
                        <code className="bg-gray-100 px-1 rounded mr-2 text-xs">cat; OCR:"food"</code>
                        <span>Multi-query</span>
                      </li>
                    </ul>
                  </div>
                </div>
                
                {/* Quick OCR Search Buttons */}
                <div className="mt-3 flex flex-wrap gap-2">
                  <span className="text-xs text-gray-500">Quick OCR:</span>
                  {['menu', 'price', 'name', 'address', 'phone', 'sign'].map((term) => (
                    <button
                      key={term}
                      type="button"
                      className="text-xs px-3 py-1.5 bg-purple-100 text-purple-700 rounded-xl hover:bg-purple-200 transition-colors duration-200 font-medium shadow-sm hover:shadow-md"
                      onClick={() => {
                        const searchBar = document.querySelector("#search-bar");
                        searchBar.value = `OCR:"${term}"`;
                        searchBar.focus();
                      }}
                    >
                      {term}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </Form>

          {/* Navigation bar for text search results */}
          <div
            id="nav-bar"
            className="p-2 flex flex-row justify-center items-center text-xl font-bold bg-white border-b border-gray-200"
          >
            <img
              onClick={() => {
                goToFirstPage();
              }}
              className="hover:bg-gray-200 active:bg-gray-300 cursor-pointer rounded-xl p-2 transition-all duration-200 hover:shadow-sm"
              width="50em"
              src={HomeButton}
              draggable="false"
            />

            <img
              onClick={() => {
                goToPreviousPage();
              }}
              className="hover:bg-gray-200 active:bg-gray-300 cursor-pointer rounded-xl p-2 transition-all duration-200 hover:shadow-sm"
              width="50em"
              src={PreviousButton}
              draggable="false"
            />
            <div className="w-12 text-center bg-gray-100 rounded-xl py-2 px-3 font-medium text-gray-700">{Math.floor(offset / limit) + 1}</div>
            <img
              onClick={() => {
                goToNextPage();
              }}
              className="hover:bg-gray-200 active:bg-gray-300 cursor-pointer rounded-xl p-2 transition-all duration-200 hover:shadow-sm"
              width="50em"
              src={NextButton}
              draggable="false"
            />
          </div>
        </div>
      )}

      {/* Audio Search Section */}
      {activeSearchTab === 'audio' && (
        <div className="flex-1 overflow-auto p-4">
          <AudioSearch
            onResults={(results) => {
              // Navigate to search results with audio query
              if (results && results.params && results.params.query) {
                submit({
                  q: `audio:"${results.params.query}"`,
                  model: "audio",
                  ...params
                }, { action: "/search" });
                setActiveSearchTab('text'); // Switch to text tab to show results
              }
            }}
            isLoading={isLoading}
            setIsLoading={() => {}} // Audio search handles its own loading state
          />
        </div>
      )}

      {/* Results Section - only show when on text search tab */}
      {activeSearchTab === 'text' && (
        <div className="flex-1 min-h-0 flex flex-col">
          {/* Results info and controls */}
          <div className="flex justify-between items-center p-2 bg-gray-50 border-b">
            <div className="text-sm text-gray-600">
              {empty ? "No results found" : `Showing ${frames.length} result${frames.length !== 1 ? 's' : ''}`}
            </div>
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={useVirtualScroll}
                onChange={(e) => setUseVirtualScroll(e.target.checked)}
                className="form-checkbox"
              />
              <span className="text-sm text-gray-600">Virtual Scroll</span>
            </label>
          </div>
          
          {/* Results content */}
          {empty ? (
            <div className="p-8 text-center flex-1">
              <div className="max-w-md mx-auto">
                <svg className="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <h3 className="text-lg font-medium text-gray-900 mb-2">No results found</h3>
                <p className="text-gray-500 mb-4">
                  {q ? `No results found for "${q}"` : "Enter a search query to find video frames"}
                </p>
                <div className="text-sm text-gray-400">
                  <p>Try:</p>
                  <ul className="list-disc list-inside mt-2 space-y-1">
                    <li>Different keywords</li>
                    <li>OCR search with OCR:"text"</li>
                    <li>Video filter with video:id</li>
                    <li>Audio search with text description</li>
                  </ul>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex-1 min-h-0">
              {useVirtualScroll && frames.length > 20 ? (
                <VirtualGrid
                  items={frames}
                  itemHeight={280}
                  itemWidth={240}
                  containerHeight={600}
                  gap={8}
                  renderItem={renderFrameItem}
                  onLoadMore={loadMoreData}
                  hasNextPage={hasNextPage}
                  isLoadingMore={isLoading}
                  className={classNames("px-4", {
                    "opacity-75": isLoading,
                  })}
                />
              ) : (
                <FrameContainer 
                  id="result" 
                  isLoading={isLoading}
                  className={classNames("", {
                    "animate-pulse": isLoading,
                  })}
                >
                  {frames.map((frame) => (
                    <FrameItem
                      key={frame.id}
                      video_id={frame.video_id}
                      frame_id={frame.frame_id}
                      thumbnail={frame.frame_uri}
                      onPlay={() => handleOnPlay(frame)}
                      onSearchSimilar={() => handleOnSearchSimilar(frame)}
                      onSelect={() => handleOnSelect(frame)}
                      selected={selectedFrame === frame.id}
                    />
                  ))}
                </FrameContainer>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
