"use client"

import { useRef, useEffect } from "react"
import { useVideoStore } from "@/lib/video-store"
import { Card } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"

export function VideoPlayer() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const { videoUrl, isVideoLoaded, currentTimestamp, setCurrentTimestamp } = useVideoStore()

  // Handle seeking to a specific timestamp
  useEffect(() => {
    if (videoRef.current && currentTimestamp !== null) {
      videoRef.current.currentTime = currentTimestamp
      videoRef.current.play().catch((err) => console.error("Error playing video:", err))
    }
  }, [currentTimestamp])

  // Update current timestamp as video plays
  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTimestamp(videoRef.current.currentTime)
    }
  }

  if (!isVideoLoaded) {
    return (
      <Card className="w-full aspect-video flex items-center justify-center bg-gray-100 dark:bg-gray-800">
        <Skeleton className="w-full h-full" />
      </Card>
    )
  }

  return (
    
    <Card className="w-full overflow-hidden">
<video
  ref={videoRef}
  src={videoUrl ?? undefined}  // Convert null to undefined.
  className="w-full aspect-video"
  controls
  onTimeUpdate={handleTimeUpdate}
/>
    </Card>
  )
}

