const APPSYNC_ENDPOINT = process.env.NEXT_PUBLIC_APPSYNC_ENDPOINT || "";
const APPSYNC_API_KEY = process.env.NEXT_PUBLIC_APPSYNC_API_KEY || "";

const SUBSCRIPTION_QUERY = JSON.stringify({
  query: `subscription OnComment {
    onComment {
      id
      content
      color
      createdAt
    }
  }`,
});

export interface Comment {
  id: string;
  content: string;
  color: string | null;
  createdAt: string;
}

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

export function subscribeToComments(
  onComment: (comment: Comment) => void,
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
        const comment = data.payload?.data?.onComment;
        if (comment) {
          onComment(comment);
        }
        break;
      }

      case "error":
        onError(data.payload);
        break;

      case "ka":
        break;
    }
  };

  ws.onerror = (err) => onError(err);
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

export async function sendComment(content: string, color: string) {
  const res = await fetch(APPSYNC_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": APPSYNC_API_KEY,
    },
    body: JSON.stringify({
      query: `mutation SendComment($content: String!, $color: String) {
        sendComment(content: $content, color: $color) {
          id content color createdAt
        }
      }`,
      variables: { content, color },
    }),
  });
  return res.json();
}
