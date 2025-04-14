import { VideoUploader } from "@/components/video-uploader"
import { SearchInterface } from "@/components/search-interface"
import { TimestampResults } from "@/components/timestamp-results"
import { VideoPlayer } from "@/components/video-player"

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center p-4 md:p-8 bg-gray-50 dark:bg-gray-900">
      <div className="w-full max-w-5xl flex flex-col items-center gap-8">
        <h1 className="text-3xl font-bold text-center mt-8 mb-2 text-gray-800 dark:text-gray-100">
          Video Content Search
        </h1>
        <p className="text-center text-gray-600 dark:text-gray-300 mb-8 max-w-2xl">
          Upload your video and use natural language to search for specific moments and content
        </p>

        <VideoUploader />

        <div className="w-full mt-6">
          <VideoPlayer />
        </div>

        <div className="w-full mt-6">
          <TimestampResults />
        </div>

        <div className="w-full mt-auto sticky bottom-0 pb-4">
          <SearchInterface />
        </div>
      </div>
    </main>
  )
}

