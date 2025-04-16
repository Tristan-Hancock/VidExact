"use client";

import React, { useState } from "react";
import { Send } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useVideoStore } from "@/lib/video-store";
import { useToast } from "@/components/ui/use-toast";

export function SearchInterface() {
  const [query, setQuery] = useState("");
  const { toast } = useToast();
  const { isVideoLoaded, isAnalyzing, addSearchResult } = useVideoStore();

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    if (!isVideoLoaded) {
      toast({
        title: "No video uploaded",
        description: "Please upload a video first.",
        variant: "destructive",
      });
      return;
    }

    if (isAnalyzing) {
      toast({
        title: "Video is being analyzed",
        description: "Please wait until analysis is complete.",
        variant: "destructive",
      });
      return;
    }

    try {
      
      console.log("Initiating search request with query:", query);
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query, top_k: 5 }),
      });

      console.log("Received response from backend with status:", response.status);
      if (!response.ok) {
        const errData = await response.json();
        console.error("Search error response:", errData);
        throw new Error(errData.error || "Search failed");
      }

      const data = await response.json();
      console.log("Search result received:", data);

      // if (data.results && Array.isArray(data.results)) {
      //   data.results.forEach((result: any) => {
      //     addSearchResult(result);
      //   });
      // } else if (data.nlp_output) {
      //   addSearchResult({
      //     id: `search-${Date.now()}`,
      //     timestamp: 0,
      //     formattedTime: "0:00",
      //     text: data.nlp_output,
      //     confidence: 1.0,
      //   });
      // }
      
      toast({
        title: "Search complete",
        description: `Found results for "${query}".`,
      });
      console.log("Received response from backend with status:", response.status);
      console.log("Search result received:", data);

    } catch (error: any) {
      console.error("Error during search:", error);
      toast({
        title: "Search error",
        description: error.message,
        variant: "destructive",
      });
    }
    setQuery("");
  };

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
  );
}
