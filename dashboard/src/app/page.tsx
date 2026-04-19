"use client";

import { useEffect, useRef, useState } from "react";
import {
  subscribeToComments,
  sendComment,
  type Comment,
} from "@/lib/appsync-realtime";

const COLORS = [
  { label: "白", value: "#FFFFFF" },
  { label: "赤", value: "#FF4444" },
  { label: "緑", value: "#44FF44" },
  { label: "青", value: "#4488FF" },
  { label: "黄", value: "#FFFF44" },
  { label: "ピンク", value: "#FF88CC" },
];

export default function Home() {
  const [text, setText] = useState("");
  const [color, setColor] = useState("#FFFFFF");
  const [sending, setSending] = useState(false);
  const [connected, setConnected] = useState(false);
  const [comments, setComments] = useState<Comment[]>([]);
  const feedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const sub = subscribeToComments(
      (comment) => {
        setComments((prev) => [comment, ...prev].slice(0, 100));
      },
      () => setConnected(true),
      (err) => console.error("Subscription error:", err)
    );
    return () => sub.close();
  }, []);

  useEffect(() => {
    feedRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  }, [comments]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim() || sending) return;
    setSending(true);
    try {
      await sendComment(text.trim(), color);
      setText("");
    } catch (err) {
      console.error("送信失敗:", err);
    }
    setSending(false);
  };

  return (
    <main className="flex flex-col items-center min-h-screen p-4 gap-6">
      <div className="w-full max-w-md space-y-6 mt-8">
        <div className="text-center">
          <span className="text-5xl">🐧</span>
          <h1 className="text-2xl font-bold mt-2">コメント</h1>
          <div className="flex items-center justify-center gap-2 mt-2 text-sm">
            <span
              className={`inline-block w-2 h-2 rounded-full ${
                connected ? "bg-green-400 animate-pulse" : "bg-gray-600"
              }`}
            />
            <span className="text-gray-400">
              {connected ? "接続中" : "未接続"}
            </span>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="コメントを入力..."
            maxLength={50}
            className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white text-lg placeholder-gray-500 focus:outline-none focus:border-blue-500"
            autoFocus
          />

          <div className="flex gap-2 justify-center">
            {COLORS.map((c) => (
              <button
                key={c.value}
                type="button"
                onClick={() => setColor(c.value)}
                className={`w-10 h-10 rounded-full border-2 transition-transform ${
                  color === c.value
                    ? "border-white scale-110"
                    : "border-gray-600"
                }`}
                style={{ backgroundColor: c.value }}
                title={c.label}
              />
            ))}
          </div>

          <button
            type="submit"
            disabled={!text.trim() || sending}
            className="w-full py-3 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white font-bold text-lg transition-colors"
          >
            {sending ? "送信中..." : "送信"}
          </button>
        </form>
      </div>

      <div className="w-full max-w-md flex-1 min-h-0">
        <h2 className="text-sm font-semibold text-gray-400 mb-2">
          コメント一覧
        </h2>
        <div
          ref={feedRef}
          className="h-80 overflow-y-auto space-y-2 rounded-xl bg-gray-900 border border-gray-800 p-3"
        >
          {comments.length === 0 ? (
            <p className="text-gray-600 text-center text-sm py-8">
              まだコメントがありません
            </p>
          ) : (
            comments.map((c) => (
              <div key={c.id} className="animate-fade-in flex items-start gap-2">
                <span
                  className="inline-block w-3 h-3 rounded-full mt-1 shrink-0"
                  style={{ backgroundColor: c.color || "#FFFFFF" }}
                />
                <span className="text-sm break-all">{c.content}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </main>
  );
}
