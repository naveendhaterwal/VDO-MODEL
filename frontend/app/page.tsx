"use client";

import { useState, useEffect } from "react";
import { Loader2, Film, PlayCircle, Download } from "lucide-react";

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt) return;

    setStatus("submitting");
    setJobId(null);
    setVideoUrl(null);

    try {
      const res = await fetch("http://localhost:8000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      const data = await res.json();
      setJobId(data.job_id);
      setStatus("queued");
    } catch (err) {
      console.error(err);
      setStatus("failed");
    }
  };

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (jobId && (status === "queued" || status === "started" || status === "deferred")) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`http://localhost:8000/status/${jobId}`);
          const data = await res.json();
          setStatus(data.status);
          if (data.status === "finished" && data.result) {
            setVideoUrl(`http://localhost:8000${data.result}`);
          }
        } catch (err) {
          console.error(err);
        }
      }, 5000); // poll every 5 seconds
    }
    return () => clearInterval(interval);
  }, [jobId, status]);

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      <form onSubmit={handleSubmit} className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-xl">
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Story Prompt
        </label>
        <textarea
          className="w-full bg-gray-900 border border-gray-600 rounded-lg p-4 text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
          rows={4}
          placeholder="A warrior returns home after years of war..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <button
          type="submit"
          disabled={status === "queued" || status === "started" || !prompt}
          className="mt-4 w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 px-4 rounded-lg flex items-center justify-center transition-colors"
        >
          {status === "queued" || status === "started" ? (
            <>
              <Loader2 className="animate-spin mr-2" size={20} />
              Generating Cinematic (This will take several minutes...)
            </>
          ) : (
            <>
              <Film className="mr-2" size={20} />
              Generate Movie
            </>
          )}
        </button>
      </form>

      {status && (
        <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-xl">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <PlayCircle className="mr-2 text-emerald-400" />
            Generation Status: <span className="ml-2 uppercase text-sm bg-gray-700 px-3 py-1 rounded-full">{status}</span>
          </h2>
          
          {videoUrl && (
            <div className="space-y-4">
              <video 
                src={videoUrl} 
                controls 
                className="w-full rounded-lg border border-gray-600 bg-black aspect-video"
                autoPlay
                loop
              />
              <a
                href={videoUrl}
                download
                className="flex items-center justify-center w-full bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors"
              >
                <Download className="mr-2" size={20} />
                Download Final Render
              </a>
            </div>
          )}
          
          {status === "failed" && (
            <div className="text-red-400 bg-red-900/20 p-4 rounded-lg border border-red-800">
              Pipeline execution failed. Please check the worker logs.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
