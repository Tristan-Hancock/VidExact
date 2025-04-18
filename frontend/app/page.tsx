import { VideoUploader } from "@/components/video-uploader"
import { SearchInterface } from "@/components/search-interface"
import { TimestampResults } from "@/components/timestamp-results"
import { VideoPlayer } from "@/components/video-player"

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center p-4 md:p-8 bg-gray-50 dark:bg-gray-900 relative">
      {/* Spinning Gradient Background Circles */}
      <div className="orange-gradient-1 absolute -right-[150px] top-[370px] -z-[1] h-[500px] w-[500px] animate-spin rounded-[500px]"></div>
      <div className="orange-gradient-2 absolute right-[57px] top-[620px] -z-[1] h-[450px] w-[450px] animate-spin rounded-[450px]"></div>

      <div className="w-full max-w-5xl flex flex-col items-center gap-8">
        {/* Logo Image */}
        <h1 className="text-3xl font-bold text-center mt-8 mb-2 text-gray-800 dark:text-gray-100">
          <img src="/assets/videxact.png" alt="VidExact Logo" className="w-full h-auto max-w-[300px]" />
        </h1>

        {/* Subtitle */}
        <p className="text-center text-gray-600 dark:text-gray-300 mb-8 max-w-2xl">
          Upload your video and use natural language to search for specific moments and content
        </p>

        {/* Video Uploader */}
        <VideoUploader />

        {/* Video Player */}
        <div className="w-full mt-6">
          <VideoPlayer />
        </div>

        {/* Timestamp Results */}
        <div className="w-full mt-6">
          <TimestampResults />
        </div>

        {/* Search Interface */}
        <div className="w-full mt-auto sticky bottom-0 pb-4">
          <SearchInterface />
        </div>
      </div>
    </main>
  )
}
