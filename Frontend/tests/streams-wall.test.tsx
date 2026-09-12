import { describe, expect, it } from "vitest";
import {
  activeStreamIds,
  createVisibleStreamState,
  moveVisibleStreamPage,
} from "../src/hooks/useVisibleStreams";

const cameraIds = Array.from(
  { length: 100 },
  (_, index) => `camera-${String(index + 1).padStart(3, "0")}`,
);

describe("streams wall subscription state", () => {
  it("keeps a 16-tile page within a 20-stream hard budget", () => {
    const state = createVisibleStreamState(cameraIds, 16, 1);

    expect(state.page).toBe(1);
    expect(state.pageCount).toBe(7);
    expect(state.visibleIds).toEqual(cameraIds.slice(0, 16));
    expect(state.prefetchIds).toEqual(cameraIds.slice(16, 20));
    expect(state.activeIds).toHaveLength(20);
  });

  it("replaces old subscriptions when the page changes", () => {
    const first = createVisibleStreamState(cameraIds, 16, 1);
    const second = moveVisibleStreamPage(first, cameraIds, 16, 2);

    expect(second.visibleIds).toEqual(cameraIds.slice(16, 32));
    expect(second.activeIds).toEqual(cameraIds.slice(16, 36));
    expect(second.activeIds).not.toContain(first.visibleIds[0]);
    expect(second.activeIds).toHaveLength(20);
  });

  it("clamps the current page when filtering shrinks the result", () => {
    const lastPage = createVisibleStreamState(cameraIds, 16, 7);
    const filtered = moveVisibleStreamPage(
      lastPage,
      cameraIds.slice(0, 18),
      16,
      lastPage.page,
    );

    expect(filtered.page).toBe(2);
    expect(filtered.pageCount).toBe(2);
    expect(filtered.visibleIds).toEqual(cameraIds.slice(16, 18));
    expect(filtered.activeIds).toEqual(cameraIds.slice(16, 18));
  });

  it("uses one grid row of prefetch for every density", () => {
    expect(createVisibleStreamState(cameraIds, 4, 1).activeIds).toHaveLength(6);
    expect(createVisibleStreamState(cameraIds, 9, 1).activeIds).toHaveLength(
      12,
    );
    expect(createVisibleStreamState(cameraIds, 16, 1).activeIds).toHaveLength(
      20,
    );
  });

  it("represents an empty result as one stable empty page", () => {
    const state = createVisibleStreamState([], 16, 8);

    expect(state.page).toBe(1);
    expect(state.pageCount).toBe(1);
    expect(state.visibleIds).toEqual([]);
    expect(state.activeIds).toEqual([]);
  });

  it("releases visible and prefetched subscriptions when the wall leaves view", () => {
    const state = createVisibleStreamState(cameraIds, 16, 1);

    expect(activeStreamIds(state, false)).toEqual([]);
    expect(activeStreamIds(state, true)).toEqual(cameraIds.slice(0, 20));
  });
});
