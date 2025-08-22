import { useState, useEffect, useRef, memo } from "react";
import classNames from "classnames";
import type { FrameContainerProps } from "@/types";

import PlayButton from "@/assets/play-btn.svg";
import SearchButton from "@/assets/search-btn.svg";

// Lazy loading image component
const LazyImage = memo(({ src, alt, className, ...props }) => {
  const [imageSrc, setImageSrc] = useState(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [isInView, setIsInView] = useState(false);
  const imgRef = useRef();

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          observer.disconnect();
        }
      },
      {
        rootMargin: '50px', // Load images 50px before they come into view
        threshold: 0.1
      }
    );

    if (imgRef.current) {
      observer.observe(imgRef.current);
    }

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (isInView && src && !imageSrc) {
      // Preload the image
      const img = new Image();
      img.onload = () => {
        setImageSrc(src);
        setIsLoaded(true);
      };
      img.onerror = () => {
        setIsLoaded(true); // Still set loaded to remove placeholder
      };
      img.src = src;
    }
  }, [isInView, src, imageSrc]);

  return (
    <div ref={imgRef} className={classNames("relative", className)}>
      {!isLoaded && (
        <div className="absolute inset-0 bg-gray-200 animate-pulse flex items-center justify-center">
          <div className="text-gray-400 text-sm">Loading...</div>
        </div>
      )}
      {imageSrc && (
        <img
          src={imageSrc}
          alt={alt}
          className={classNames(
            "transition-opacity duration-300",
            isLoaded ? "opacity-100" : "opacity-0",
            className
          )}
          {...props}
        />
      )}
    </div>
  );
});

LazyImage.displayName = 'LazyImage';

export const FrameItem = memo(({
  video_id,
  frame_id,
  thumbnail,
  onPlay,
  onSearchSimilar,
  onSelect,
  selected,
}) => {
  return (
    <div
      className={classNames(
        "relative basis-1/5 flex flex-col space-y-2 p-3 transition-all duration-200 rounded-2xl shadow-sm hover:shadow-md",
        {
          "bg-white hover:bg-gray-50 border border-gray-200 hover:border-gray-300": !selected,
          "bg-green-500 border-2 border-green-600 shadow-lg": selected,
        }
      )}
      onDoubleClick={onSelect}
    >
      <LazyImage
        src={thumbnail}
        alt={`Frame ${frame_id} from video ${video_id}`}
        className="w-full h-auto rounded-xl"
        draggable="false"
      />
      <div
        onClick={(e) => {
          e.stopPropagation();
        }}
        className="flex flex-row bg-white rounded-xl justify-end space-x-2 items-center p-2 shadow-sm"
      >
        <img
          onClick={onPlay}
          className="hover:bg-gray-200 active:bg-gray-300 cursor-pointer transition-all duration-150 rounded-lg p-1.5 hover:shadow-sm"
          width="30"
          height="30"
          src={PlayButton}
          alt="Play video"
          draggable="false"
        />
        <img
          onClick={onSearchSimilar}
          className="hover:bg-gray-200 active:bg-gray-300 cursor-pointer transition-all duration-150 rounded-lg p-1.5 hover:shadow-sm"
          width="30"
          height="30"
          src={SearchButton}
          alt="Search similar"
          draggable="false"
        />
      </div>
      <div className="absolute top-0 left-0 space-x-2 flex flex-row bg-black bg-opacity-60 rounded-br-md p-1">
        <div className="text-sm text-white font-mono">{frame_id}</div>
        <div className="text-sm text-nowrap overflow-hidden text-white">
          {video_id}
        </div>
      </div>
    </div>
  );
});

FrameItem.displayName = 'FrameItem';

export const FrameContainer = memo(({ children, isLoading, className = "" }: FrameContainerProps) => {
  const hasChildren = Array.isArray(children) ? children.length > 0 : !!children;
  
  return (
    <div 
      className={classNames(
        "flex flex-wrap gap-2 p-2 min-h-[200px]",
        className,
        {
          "animate-pulse": isLoading
        }
      )}
    >
      {hasChildren ? (
        children
      ) : (
        <div className="w-full flex flex-col items-center justify-center text-gray-500 py-8">
          <div className="text-center">
            <svg className="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <p className="text-sm">Frame container ready</p>
            <p className="text-xs text-gray-400 mt-1">Video frames will appear here</p>
          </div>
        </div>
      )}
    </div>
  );
});

FrameContainer.displayName = 'FrameContainer';
