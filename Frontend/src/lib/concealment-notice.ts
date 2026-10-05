export type ConcealmentNotice = {
  id: string;
  phase: "verifying" | "review" | "inconclusive";
  expires_at: number;
  created_at: number;
};

export function activeConcealmentNotice(value: unknown, now = Date.now()): ConcealmentNotice | null {
  if (!value || typeof value !== "object") return null;
  const notice = value as ConcealmentNotice;
  if (typeof notice.id !== "string" ||
      !["verifying", "review", "inconclusive"].includes(notice.phase) ||
      !Number.isFinite(notice.expires_at) || !Number.isFinite(notice.created_at) ||
      notice.expires_at * 1000 <= now || notice.created_at * 1000 > now + 1000 ||
      notice.expires_at - notice.created_at > 8) return null;
  return notice;
}
