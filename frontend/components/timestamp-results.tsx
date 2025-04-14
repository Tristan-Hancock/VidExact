"use client"

import { useVideoStore } from "@/lib/video-store"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Clock, ArrowRight } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"

export function TimestampResults() {
  const { searchResults, setCurrentTimestamp, isVideoLoaded, isAnalyzing } = useVideoStore()

  const handleJumpToTimestamp = (timestamp: number) => {
    setCurrentTimestamp(timestamp)
  }

  if (!isVideoLoaded) {
    return null
  }

  if (isAnalyzing) {
    return (
      <Card className="w-full p-6 bg-white dark:bg-gray-800">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          <p className="ml-3 text-gray-600 dark:text-gray-300">Analyzing video content...</p>
        </div>
      </Card>
    )
  }

  if (searchResults.length === 0) {
    return (
      <Card className="w-full p-6 bg-white dark:bg-gray-800">
        <p className="text-center text-gray-500 dark:text-gray-400">Search for content to see timestamps and results</p>
      </Card>
    )
  }

  return (
    <Card className="w-full p-4 bg-white dark:bg-gray-800">
      <h3 className="text-lg font-medium mb-4">Search Results</h3>

      <ScrollArea className="h-[250px] pr-4">
        <div className="space-y-3">
          {searchResults.map((result) => (
            <div
              key={result.id}
              className="p-3 border rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3">
                  <div className="mt-1">
                    <Clock className="h-5 w-5 text-gray-500 dark:text-gray-400" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <Badge variant="outline" className="font-mono">
                        {result.formattedTime}
                      </Badge>
                      <Badge
                        variant="secondary"
                        className={
                          result.confidence > 0.9
                            ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-100"
                            : "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-100"
                        }
                      >
                        {Math.round(result.confidence * 100)}% match
                      </Badge>
                    </div>
                    <p className="text-gray-700 dark:text-gray-300">{result.text}</p>
                  </div>
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => handleJumpToTimestamp(result.timestamp)}
                  className="ml-2 shrink-0"
                >
                  <ArrowRight className="h-4 w-4 mr-1" />
                  Jump
                </Button>
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>
    </Card>
  )
}

