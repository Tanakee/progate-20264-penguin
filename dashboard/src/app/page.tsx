"use client";

import { useEffect, useRef, useState } from "react";
import { subscribeToEvents } from "@/lib/appsync-realtime";
import type { ClapEvent } from "@/lib/types";
import EventFeed from "@/components/EventFeed";
import ExcitementChart from "@/components/ExcitementChart";
import EmotionSummary from "@/components/EmotionSummary";
import StatsBar from "@/components/StatsBar";
import SessionSummary from "@/components/SessionSummary";

type ViewMode = "live" | "summary";

export default function Home() {
  const [events, setEvents] = useState<ClapEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>("live");
  const subRef = useRef<{ close: () => void } | null>(null);

  const isConfigured = !!process.env.NEXT_PUBLIC_APPSYNC_ENDPOINT;

  useEffect(() => {
    if (!isConfigured) return;

    const sub = subscribeToEvents(
      (event) => {
        if (event.id === "__summary__") {
          setViewMode("summary");
          return;
        }
        setEvents((prev) => [event, ...prev].slice(0, 100));
      },
      () => {
        setConnected(true);
        console.log("AppSync Subscription connected");
      },
      (err) => {
        console.error("Subscription error:", err);
        setConnected(false);
      }
    );
    subRef.current = sub;

    return () => {
      sub.close();
    };
  }, [isConfigured]);

  if (viewMode === "summary") {
    return (
      <SessionSummary
        events={events}
        onReset={() => {
          setEvents([]);
          setViewMode("live");
        }}
      />
    );
  }

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🐧</span>
            <h1 className="text-xl font-bold">ファースト肯定ペンギン</h1>
          </div>
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                connected ? "bg-green-500 animate-pulse" : "bg-gray-600"
              }`}
            />
            <span className="text-sm text-gray-400">
              {connected
                ? "リアルタイム接続中"
                : isConfigured
                  ? "接続中..."
                  : "デモモード"}
            </span>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        {!isConfigured && (
          <div className="bg-yellow-900/30 border border-yellow-700/50 rounded-xl p-4 text-yellow-200 text-sm">
            AppSync 未設定のためデモモードで表示中。
            <code className="ml-1 text-xs bg-yellow-900/50 px-1 rounded">
              NEXT_PUBLIC_APPSYNC_ENDPOINT
            </code>{" "}
            を設定してください。
          </div>
        )}

        {/* Stats */}
        <StatsBar events={events} />

        {/* Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Event Feed */}
          <div className="lg:col-span-2 bg-gray-900 rounded-xl border border-gray-800 p-6">
            <h2 className="text-lg font-semibold mb-4">拍手イベント</h2>
            <EventFeed events={events} />
          </div>

          {/* Emotion Summary */}
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
            <h2 className="text-lg font-semibold mb-4">感情分析</h2>
            <EmotionSummary events={events} />
          </div>
        </div>

        {/* Chart */}
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
          <h2 className="text-lg font-semibold mb-4">会場の盛り上がり</h2>
          <ExcitementChart events={events} />
        </div>
      </main>
    </div>
  );
}
