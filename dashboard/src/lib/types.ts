export interface Emotion {
  type: string;
  confidence: number;
}

export interface ClapEvent {
  id: string;
  timestamp: string;
  trackId: number;
  imageUrl: string;
  composedImageUrl?: string;
  emotions?: Emotion[];
  dominantEmotion?: string;
  confidence?: number;
}
