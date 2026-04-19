"use client";

import { useState } from "react";

const APPSYNC_ENDPOINT = process.env.NEXT_PUBLIC_APPSYNC_ENDPOINT || "";
const APPSYNC_API_KEY = process.env.NEXT_PUBLIC_APPSYNC_API_KEY || "";

const COLORS = [
  { label: "白", value: "#FFFFFF" },
  { label: "赤", value: "#FF4444" },
  { label: "緑", value: "#44FF44" },
  { label: "青", value: "#4488FF" },
  { label: "黄", value: "#FFFF44" },
  { label: "ピンク", value: "#FF88CC" },
];

async function sendComment(text: string, color: string) {
  const mutation = `
    mutation PublishComment($input: CommentInput!) {
      publishComment(input: $input) {
        id
        text
        color
        timestamp
      }
    }
  `;

  const variables = {
    input: {
      id: crypto.randomUUID(),
      text,
      color,
      timestamp: new Date().toISOString(),
    },
  };

  const res = await fetch(APPSYNC_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": APPSYNC_API_KEY,
    },
    body: JSON.stringify({ query: mutation, variables }),
  });

  return res.json();
}

export default function CommentPage() {
  const [text, setText] = useState("");
  const [color, setColor] = useState("#FFFFFF");
  const [sending, setSending] = useState(false);

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
    <div className="min-h-screen bg-gray-950 flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-sm space-y-6">
        <div className="text-center">
          <span className="text-4xl">🐧</span>
          <h1 className="text-xl font-bold mt-2">コメント投稿</h1>
          <p className="text-sm text-gray-400 mt-1">画面にコメントが流れます</p>
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
    </div>
  );
}
