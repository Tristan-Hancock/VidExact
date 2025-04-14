"use client"

import type React from "react"

import { useState } from "react"
import { Upload, Video } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useToast } from "@/components/ui/use-toast"
import { useVideoStore } from "@/lib/video-store"

export function VideoUploader() {
  const { toast } = useToast()
  const { setVideo, setIsAnalyzing, setIsVideoLoaded } = useVideoStore()
  const [isDragging, setIsDragging] = useState(false)

  const handleFileUpload = (file: File) => {
    // Check if the file is a video
    if (!file.type.startsWith("video/")) {
      toast({
        title: "Invalid file type",
        description: "Please upload a video file.",
        variant: "destructive",
      })
      return
    }

    // Create a URL for the video
    const videoUrl = URL.createObjectURL(file)
    setVideo(videoUrl)
    setIsVideoLoaded(true)

    // Simulate video analysis
    setIsAnalyzing(true)
    toast({
      title: "Video uploaded",
      description: "Analyzing video content...",
    })

    // Simulate analysis completion after 3 seconds
    setTimeout(() => {
      setIsAnalyzing(false)
      toast({
        title: "Analysis complete",
        description: "Your video is ready to be searched.",
      })
    }, 3000)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0])
    }
  }

  return (
    <div
      className={`w-full max-w-xl mx-auto border-2 border-dashed rounded-lg p-8 flex flex-col items-center justify-center transition-colors ${
        isDragging ? "border-primary bg-primary/5" : "border-gray-300 dark:border-gray-700 hover:border-primary/50"
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="flex flex-col items-center text-center">
        <div className="h-20 w-20 rounded-full bg-primary/10 flex items-center justify-center mb-4">
          <Video className="h-10 w-10 text-primary" />
        </div>
        <h3 className="text-lg font-medium mb-2">Upload your video</h3>
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
          Drag and drop your video file here, or click to browse
        </p>

        <label htmlFor="video-upload">
          <Button className="flex items-center gap-2">
            <Upload className="h-4 w-4" />
            Select Video
          </Button>
          <input
            id="video-upload"
            type="file"
            accept="video/*"
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files.length > 0) {
                handleFileUpload(e.target.files[0])
              }
            }}
          />
        </label>
      </div>
    </div>
  )
}

