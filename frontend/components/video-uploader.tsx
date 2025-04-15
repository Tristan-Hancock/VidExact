"use client";

import React, { useState, useRef } from "react";
import { Upload, Video } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/use-toast";
import { useVideoStore } from "@/lib/video-store";

export function VideoUploader() {
  const { toast } = useToast();
  const { setVideo, setIsAnalyzing, setIsVideoLoaded } = useVideoStore();
  const [isDragging, setIsDragging] = useState(false);

  // Create a ref for the file input
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Function to programmatically trigger the file input click
  const handleButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleFileUpload = async (file: File) => {
    console.log("Starting file upload process...");
    // Check if the file is a valid video format.
    if (!file.type.endsWith("mp4") && !file.type.endsWith("webm")) {
      toast({
        title: "Invalid file type",
        description: "Please upload a video file.",
        variant: "destructive",
      });
      console.log("File type invalid:", file.type);
      return;
    }

    // Log file details for debugging.
    console.log("Uploading file:", file.name, file.type, file.size);

    // Create a FormData object to send the file via POST.
    const formData = new FormData();
    formData.append("file", file);

    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;
      console.log("Uploading to:", backendUrl + "/api/upload");

      const response = await fetch(`${backendUrl}/api/upload`, {
        method: "POST",
        body: formData,
      });
      
      console.log("Received response from backend:", response.status);

      if (!response.ok) {
        const errData = await response.json();
        console.error("Upload error response:", errData);
        throw new Error(errData.error || "Upload failed");
      }
      const result = await response.json();
      console.log("File upload successful. Server response:", result);

      toast({
        title: "Video uploaded",
        description: "Video has been successfully uploaded.",
      });
      // Set the local video URL for preview (if desired).
      setVideo(URL.createObjectURL(file));
      setIsVideoLoaded(true);

      // Simulate further processing.
      console.log("Starting video analysis simulation...");
      setIsAnalyzing(true);
      setTimeout(() => {
        setIsAnalyzing(false);
        toast({
          title: "Analysis complete",
          description: "Your video is ready to be searched.",
        });
        console.log("Video analysis simulation complete.");
      }, 3000);
    } catch (error: any) {
      console.error("Error during file upload:", error);
      toast({
        title: "Upload error",
        description: error.message,
        variant: "destructive",
      });
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      console.log("File dropped:", e.dataTransfer.files[0].name);
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  return (
    <div
      className={`w-full max-w-xl mx-auto border-2 border-dashed rounded-lg p-8 flex flex-col items-center justify-center transition-colors ${
        isDragging
          ? "border-primary bg-primary/5"
          : "border-gray-300 dark:border-gray-700 hover:border-primary/50"
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
          Drag and drop your video file here, or click to browse.
        </p>

        {/* Remove htmlFor and onChange from the label since we handle click via ref */}
        <Button onClick={handleButtonClick} className="flex items-center gap-2">
          <Upload className="h-4 w-4" />
          Select Video
        </Button>
        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          id="video-upload"
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              console.log("Selected file:", e.target.files[0].name);
              handleFileUpload(e.target.files[0]);
            }
          }}
        />
      </div>
    </div>
  );
}
