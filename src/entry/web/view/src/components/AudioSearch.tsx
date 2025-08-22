import { useState, memo, useCallback, KeyboardEvent } from 'react';
import { searchByAudioText } from '@/services/search';
import type { AudioSearchProps } from '@/types';

const AudioSearch = memo(({ onResults, isLoading, setIsLoading }: AudioSearchProps) => {
  const [audioQuery, setAudioQuery] = useState('');
  const [searchParams, setSearchParams] = useState({
    offset: 0,
    limit: 50,
    nprobe: 8,
    model: "audio"
  });

  const handleSearch = useCallback(async () => {
    if (!audioQuery.trim()) {
      alert('Please enter a search query for audio content');
      return;
    }

    setIsLoading(true);
    try {
      const results = await searchByAudioText(
        audioQuery.trim(),
        searchParams.offset,
        searchParams.limit,
        searchParams.nprobe,
        searchParams.model
      );
      onResults(results);
    } catch (error) {
      console.error('Audio text search failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      alert('Audio text search failed: ' + errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [audioQuery, searchParams, setIsLoading, onResults]);

  const handleKeyDown = useCallback((e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSearch().then();
    }
  }, [handleSearch]);

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-4">
      <h3 className="text-lg font-semibold mb-4 flex items-center">
        <svg className="w-6 h-6 mr-2 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M9 12a3 3 0 106 0 3 3 0 00-6 0z" />
        </svg>
        Audio Search
      </h3>
      
      {/* Text Input Area */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Search for audio content by description
        </label>
        <div className="relative">
          <textarea
            value={audioQuery}
            onChange={(e) => setAudioQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe the audio you're looking for (e.g., 'music playing', 'people talking', 'car engine', 'applause')"
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            rows={3}
          />
          <div className="absolute bottom-2 right-2 text-xs text-gray-400">
            Press Enter to search
          </div>
        </div>
        <div className="mt-2 text-sm text-gray-500">
          <p className="mb-1">Examples:</p>
          <div className="flex flex-wrap gap-2">
            {[
              'music playing',
              'people talking', 
              'car engine',
              'applause',
              'door closing',
              'phone ringing'
            ].map((example) => (
              <button
                key={example}
                onClick={() => setAudioQuery(example)}
                className="px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
                type="button"
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Search Parameters */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Limit
          </label>
          <select
            value={searchParams.limit}
            onChange={(e) => setSearchParams(prev => ({ ...prev, limit: parseInt(e.target.value) }))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value={20}>20</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            nProbe
          </label>
          <select
            value={searchParams.nprobe}
            onChange={(e) => setSearchParams(prev => ({ ...prev, nprobe: parseInt(e.target.value) }))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value={4}>4</option>
            <option value={8}>8</option>
            <option value={16}>16</option>
            <option value={32}>32</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Model
          </label>
          <select
            value={searchParams.model}
            onChange={(e) => setSearchParams(prev => ({ ...prev, model: e.target.value }))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="audio">Audio</option>
          </select>
        </div>

        <div className="flex items-end">
          <button
            onClick={handleSearch}
            disabled={!audioQuery.trim() || isLoading}
            className={`w-full px-4 py-2 rounded-md font-medium transition-colors duration-200 ${
              !audioQuery.trim() || isLoading
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-blue-600 text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500'
            }`}
          >
            {isLoading ? (
              <div className="flex items-center justify-center">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                Searching...
              </div>
            ) : (
              'Search Audio'
            )}
          </button>
        </div>
      </div>
    </div>
  );
});

AudioSearch.displayName = 'AudioSearch';

export default AudioSearch;