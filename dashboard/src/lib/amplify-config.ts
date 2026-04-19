const amplifyConfig = {
  API: {
    GraphQL: {
      endpoint: process.env.NEXT_PUBLIC_APPSYNC_ENDPOINT || "",
      defaultAuthMode: "apiKey" as const,
      apiKey: process.env.NEXT_PUBLIC_APPSYNC_API_KEY || "",
      region: process.env.NEXT_PUBLIC_AWS_REGION || "ap-northeast-1",
    },
  },
};

export function getRealtimeUrl(): string {
  const httpUrl = process.env.NEXT_PUBLIC_APPSYNC_ENDPOINT || "";
  return httpUrl
    .replace("https://", "wss://")
    .replace("appsync-api", "appsync-realtime-api");
}

export default amplifyConfig;
