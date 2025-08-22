import { useState, useEffect, useRef, useCallback, memo } from 'react';

const VirtualGrid = memo(({
  items = [],
  itemHeight = 200,
  itemWidth = 200,
  containerHeight = 600,
  gap = 8,
  renderItem,
  className = "",
  onLoadMore = null,
  hasNextPage = false,
  isLoadingMore = false
}) => {
  const [scrollTop, setScrollTop] = useState(0);
  const [containerWidth, setContainerWidth] = useState(0);
  const containerRef = useRef(null);

  // Calculate grid dimensions
  const itemsPerRow = Math.floor((containerWidth + gap) / (itemWidth + gap)) || 1;
  const totalRows = Math.ceil(items.length / itemsPerRow);
  const totalHeight = totalRows * (itemHeight + gap);

  // Calculate visible range
  const startRow = Math.floor(scrollTop / (itemHeight + gap));
  const endRow = Math.min(
    startRow + Math.ceil(containerHeight / (itemHeight + gap)) + 1,
    totalRows
  );

  const visibleItems = [];
  for (let row = startRow; row < endRow; row++) {
    for (let col = 0; col < itemsPerRow; col++) {
      const index = row * itemsPerRow + col;
      if (index < items.length) {
        const x = col * (itemWidth + gap);
        const y = row * (itemHeight + gap);
        visibleItems.push({
          index,
          item: items[index],
          style: {
            position: 'absolute',
            left: x,
            top: y,
            width: itemWidth,
            height: itemHeight,
          },
        });
      }
    }
  }

  // Handle scroll
  const handleScroll = useCallback((e) => {
    const scrollTop = e.target.scrollTop;
    setScrollTop(scrollTop);

    // Load more when near bottom
    if (
      onLoadMore &&
      hasNextPage &&
      !isLoadingMore &&
      scrollTop + containerHeight > totalHeight - itemHeight * 2
    ) {
      onLoadMore();
    }
  }, [onLoadMore, hasNextPage, isLoadingMore, totalHeight, containerHeight, itemHeight]);

  // Update container width on resize
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        setContainerWidth(containerRef.current.clientWidth);
      }
    };

    updateWidth();
    window.addEventListener('resize', updateWidth);
    return () => window.removeEventListener('resize', updateWidth);
  }, []);

  // Create intersection observer for loading more items
  const loadMoreRef = useRef(null);
  useEffect(() => {
    if (!onLoadMore || !hasNextPage || isLoadingMore) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          onLoadMore();
        }
      },
      { threshold: 0.1 }
    );

    if (loadMoreRef.current) {
      observer.observe(loadMoreRef.current);
    }

    return () => observer.disconnect();
  }, [onLoadMore, hasNextPage, isLoadingMore]);

  return (
    <div className={className}>
      <div
        ref={containerRef}
        style={{ height: containerHeight, overflow: 'auto' }}
        onScroll={handleScroll}
        className="relative"
      >
        {items.length === 0 ? (
          /* Empty state for virtual grid */
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <div className="text-center">
              <svg className="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
              <p className="text-sm">Virtual grid ready</p>
              <p className="text-xs text-gray-400 mt-1">Items will appear here when loaded</p>
            </div>
          </div>
        ) : (
          <div style={{ height: totalHeight, position: 'relative' }}>
            {visibleItems.map(({ index, item, style }) => (
              <div key={item.id || index} style={style}>
                {renderItem(item, index)}
              </div>
            ))}
          </div>
        )}
        
        {/* Load more trigger */}
        {hasNextPage && items.length > 0 && (
          <div
            ref={loadMoreRef}
            className="flex justify-center items-center py-4"
            style={{ position: 'absolute', top: totalHeight - 100, width: '100%' }}
          >
            {isLoadingMore ? (
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
            ) : (
              <div className="text-gray-500">Loading more...</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
});

VirtualGrid.displayName = 'VirtualGrid';

export default VirtualGrid;