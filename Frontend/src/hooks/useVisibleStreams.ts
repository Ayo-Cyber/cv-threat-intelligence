import { useCallback, useEffect, useMemo, useState } from "react";

export interface VisibleStreamState {
  activeIds: string[];
  visibleIds: string[];
  prefetchIds: string[];
  page: number;
  pageCount: number;
}

function normalizedPageSize(pageSize: number): number {
  return [4, 9, 16].includes(pageSize) ? pageSize : 4;
}

export function createVisibleStreamState(
  cameraIds: string[],
  requestedPageSize: number,
  requestedPage: number,
): VisibleStreamState {
  const pageSize = normalizedPageSize(requestedPageSize);
  const pageCount = Math.max(1, Math.ceil(cameraIds.length / pageSize));
  const page = Math.min(Math.max(1, Math.trunc(requestedPage) || 1), pageCount);
  const start = (page - 1) * pageSize;
  const visibleIds = cameraIds.slice(start, start + pageSize);
  const rowSize = Math.sqrt(pageSize);
  const prefetchIds = cameraIds.slice(
    start + pageSize,
    start + pageSize + rowSize,
  );

  return {
    activeIds: [...visibleIds, ...prefetchIds],
    visibleIds,
    prefetchIds,
    page,
    pageCount,
  };
}

export function moveVisibleStreamPage(
  _current: VisibleStreamState,
  cameraIds: string[],
  pageSize: number,
  page: number,
): VisibleStreamState {
  return createVisibleStreamState(cameraIds, pageSize, page);
}

export function activeStreamIds(
  state: VisibleStreamState,
  wallVisible: boolean,
): string[] {
  return wallVisible ? state.activeIds : [];
}

export function useVisibleStreams(cameraIds: string[], pageSize: number) {
  const [requestedPage, setPage] = useState(1);
  const [wallVisible, setWallVisible] = useState(true);
  const [wallElement, setWallElement] = useState<HTMLElement | null>(null);
  const state = useMemo(
    () => createVisibleStreamState(cameraIds, pageSize, requestedPage),
    [cameraIds, pageSize, requestedPage],
  );

  useEffect(() => {
    if (state.page !== requestedPage) setPage(state.page);
  }, [requestedPage, state.page]);

  useEffect(() => {
    if (!wallElement || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      ([entry]) => setWallVisible(entry.isIntersecting),
      { rootMargin: "100px 0px" },
    );
    observer.observe(wallElement);
    return () => observer.disconnect();
  }, [wallElement]);

  const observeWall = useCallback((element: HTMLElement | null) => {
    setWallElement(element);
  }, []);

  return {
    ...state,
    activeIds: activeStreamIds(state, wallVisible),
    setPage,
    observeWall,
  };
}
