"use client"

import type React from "react"

import { useState } from "react"
import { Send } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { useVideoStore } from "@/lib/video-store"
import { useToast } from "@/components/ui/use-toast"

export function SearchInterface() {
  const [query, setQuery] = useState("")
  const { toast } = useToast()
  const { isVideoLoaded, isAnalyzing, addSearchResult } = useVideoStore()

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()

    if (!query.trim()) return

    if (!isVideoLoaded) {
      toast({
        title: "No video uploaded",
        description: "Please upload a video first.",
        variant: "destructive",
      })
      return
    }

    if (isAnalyzing) {
      toast({
        title: "Video is being analyzed",
        description: "Please wait until analysis is complete.",
        variant: "destructive",
      })
      return
    }

    // Simulate NLP search results
    const mockResults = generateMockResults(query)
    mockResults.forEach((result) => addSearchResult(result))

    // Clear the input
    setQuery("")

    toast({
      title: "Search complete",
      description: `Found ${mockResults.length} results for "${query}"`,
    })
  }

  // Generate mock search results based on the query
  const generateMockResults = (searchQuery: string) => {
    const keywords = searchQuery.toLowerCase().split(" ")
    const results = []

    // Create 1-3 mock results
    const resultCount = Math.floor(Math.random() * 3) + 1

    for (let i = 0; i < resultCount; i++) {
      // Generate random timestamp between 0 and 10 minutes
      const timestamp = Math.floor(Math.random() * 600)
      const minutes = Math.floor(timestamp / 60)
      const seconds = timestamp % 60

      // Create a relevant description based on the query
      const keywordToUse = keywords[Math.floor(Math.random() * keywords.length)]

      results.push({
        id: `result-${Date.now()}-${i}`,
        timestamp,
        formattedTime: `${minutes}:${seconds.toString().padStart(2, "0")}`,
        text: `Found "${keywordToUse}" at ${minutes}:${seconds.toString().padStart(2, "0")}`,
        confidence: Math.random() * 0.3 + 0.7, // Random confidence between 70% and 100%
      })
    }

    return results
  }

  return (
    <Card className="p-4 bg-white dark:bg-gray-800 shadow-lg border border-gray-200 dark:border-gray-700 rounded-lg">
      <form onSubmit={handleSearch} className="flex gap-2">
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search for content in the video..."
          className="flex-1"
          disabled={!isVideoLoaded || isAnalyzing}
        />
        <Button type="submit" disabled={!isVideoLoaded || isAnalyzing || !query.trim()}>
          <Send className="h-4 w-4" />
          <span className="sr-only">Search</span>
        </Button>
      </form>
    </Card>
  )
}

