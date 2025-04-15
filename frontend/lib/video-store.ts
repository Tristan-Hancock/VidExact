import { create } from "zustand"

type SearchResult = {
  id: string
  timestamp: number
  formattedTime: string
  text: string
  confidence: number
  clip_name?: string  // add this property as optional
}


type VideoStore = {
  videoUrl: string | null
  isVideoLoaded: boolean
  isAnalyzing: boolean
  currentTimestamp: number | null
  searchResults: SearchResult[]
  setVideo: (url: string) => void
  setIsVideoLoaded: (loaded: boolean) => void
  setIsAnalyzing: (analyzing: boolean) => void
  setCurrentTimestamp: (timestamp: number) => void
  addSearchResult: (result: SearchResult) => void
  clearSearchResults: () => void
}

export const useVideoStore = create<VideoStore>((set) => ({
  videoUrl: null,
  isVideoLoaded: false,
  isAnalyzing: false,
  currentTimestamp: null,
  searchResults: [],

  setVideo: (url) => set({ videoUrl: url }),
  setIsVideoLoaded: (loaded) => set({ isVideoLoaded: loaded }),
  setIsAnalyzing: (analyzing) => set({ isAnalyzing: analyzing }),
  setCurrentTimestamp: (timestamp) => set({ currentTimestamp: timestamp }),

  addSearchResult: (result) =>
    set((state) => ({
      searchResults: [result, ...state.searchResults],
    })),

  clearSearchResults: () => set({ searchResults: [] }),
}))

