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
  const { isVideoLoaded, isAnalyzing, addSearchResult, clearSearchResults } = useVideoStore();

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
      clearSearchResults(); // Clear previous results

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

      if (data.results && Array.isArray(data.results)) {
        data.results.forEach((result: any) => {
          // Here we only add timestamp and formattedTime.
          addSearchResult({
            timestamp: result.timestamp,
            formattedTime: result.formattedTime,
          });
        });
      } else if (data.nlp_output) {
        addSearchResult({
          timestamp: 0,
          formattedTime: "0:00",
        });
      }
      
      toast({
        title: "Search complete",
        description: `Found results for "${query}".`,
      });
      console.log("Final search result received:", data);

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
    <Card className="p-4 bg-[#F6E1C3] dark:bg-[#222222] shadow-lg border border-[#FF7A00] dark:border-[#FF7A00] rounded-lg">
      <form onSubmit={handleSearch} className="flex gap-2">
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search for content in the video..."
          className="flex-1 border-[#FF7A00] focus:border-[#FF7A00] focus:ring-2 focus:ring-[#FF7A00]"
          disabled={!isVideoLoaded || isAnalyzing}
        />
        <Button
          type="submit"
          disabled={!isVideoLoaded || isAnalyzing || !query.trim()}
          className="bg-[#FF7A00] hover:bg-[#FFB72B] text-white border-[#FF7A00] hover:border-[#FFB72B] transition-all duration-300"
        >
          <Send className="h-4 w-4" />
          <span className="sr-only">Search</span>
        </Button>
      </form>
    </Card>
  );
}
