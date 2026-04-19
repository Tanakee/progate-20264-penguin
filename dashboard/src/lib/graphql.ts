export const SUBSCRIBE_CLAP_EVENT = `
  subscription OnClapEvent {
    onClapEvent {
      id
      timestamp
      trackId
      imageUrl
      composedImageUrl
      emotions {
        type
        confidence
      }
      dominantEmotion
      confidence
    }
  }
`;

export const GET_RECENT_EVENTS = `
  query GetRecentEvents($limit: Int) {
    getRecentEvents(limit: $limit) {
      id
      timestamp
      trackId
      imageUrl
      composedImageUrl
      emotions {
        type
        confidence
      }
      dominantEmotion
      confidence
    }
  }
`;
