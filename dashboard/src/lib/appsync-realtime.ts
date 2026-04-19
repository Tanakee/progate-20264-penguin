import type { ClapEvent } from "./types";

const APPSYNC_ENDPOINT = process.env.NEXT_PUBLIC_APPSYNC_ENDPOINT || "";
const APPSYNC_API_KEY = process.env.NEXT_PUBLIC_APPSYNC_API_KEY || "";

const SUBSCRIPTION_QUERY = JSON.stringify({
  query: `subscription OnClapEvent {
    onClapEvent {
      id
      timestamp
      trackId
      imageUrl
      composedImageUrl
      emotions { type confidence }
      dominantEmotion
      confidence
    }
  }`,
});

function getRealtimeUrl(): string {
  return APPSYNC_ENDPOINT.replace("https://", "wss://").replace(
    "appsync-api",
    "appsync-realtime-api"
  );
}

function encodeHeader(): string {
  return btoa(
    JSON.stringify({
      host: new URL(APPSYNC_ENDPOINT).host,
      "x-api-key": APPSYNC_API_KEY,
    })
  );
}

export function subscribeToEvents(
  onEvent: (event: ClapEvent) => void,
  onConnect: () => void,
  onError: (err: unknown) => void
): { close: () => void } {
  if (!APPSYNC_ENDPOINT || !APPSYNC_API_KEY) {
    return { close: () => {} };
  }

  const header = encodeHeader();
  const payload = btoa("{}");
  const url = `${getRealtimeUrl()}?header=${encodeURIComponent(header)}&payload=${payload}`;

  const ws = new WebSocket(url, ["graphql-ws"]);
  let subscriptionId: string | null = null;

  ws.onopen = () => {
    ws.send(JSON.stringify({ type: "connection_init" }));
  };

  ws.onmessage = (msg) => {
    const data = JSON.parse(msg.data);
    console.log("[WS] received:", data.type, data);

    switch (data.type) {
      case "connection_ack": {
        subscriptionId = crypto.randomUUID();
        ws.send(
          JSON.stringify({
            id: subscriptionId,
            type: "start",
            payload: {
              data: SUBSCRIPTION_QUERY,
              extensions: {
                authorization: {
                  host: new URL(APPSYNC_ENDPOINT).host,
                  "x-api-key": APPSYNC_API_KEY,
                },
              },
            },
          })
        );
        break;
      }

      case "start_ack":
        onConnect();
        break;

      case "data": {
        console.log("[WS] data payload:", JSON.stringify(data.payload, null, 2));
        const event = data.payload?.data?.onClapEvent;
        if (event) {
          onEvent(event);
        }
        break;
      }

      case "error":
        console.error("AppSync WS error:", JSON.stringify(data.payload, null, 2));
        console.error("AppSync WS error full msg:", JSON.stringify(data, null, 2));
        onError(data.payload);
        break;

      case "ka":
        break;
    }
  };

  ws.onerror = (err) => {
    console.error("WebSocket error:", err);
    onError(err);
  };

  ws.onclose = () => {};

  return {
    close: () => {
      if (subscriptionId && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ id: subscriptionId, type: "stop" }));
      }
      ws.close();
    },
  };
}
