"use client";

import { useState, useEffect } from "react";

const API_BASE = "";

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([
    "System initialized. Awaiting job...",
    "> Network status: ONLINE",
    "> Connected to Nosana Compute Grid."
  ]);

  const handleSubmit = async () => {
    if (!prompt) return;

    setStatus("submitting");
    setJobId(null);
    setVideoUrl(null);
    setLogs((prev) => [...prev, `> Submitting job: ${prompt.substring(0, 30)}...`]);

    try {
      const res = await fetch(`${API_BASE}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      const data = await res.json();
      setJobId(data.job_id);
      setStatus("queued");
      setLogs((prev) => [...prev, `> Job queued. ID: ${data.job_id}`]);
    } catch (err) {
      console.error(err);
      setStatus("failed");
      setLogs((prev) => [...prev, "> ERROR: Failed to connect to compute node."]);
    }
  };

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (jobId && (status === "queued" || status === "started" || status === "deferred")) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/status/${jobId}`);
          const data = await res.json();
          
          if (status !== data.status) {
             setLogs((prev) => [...prev, `> Status update: ${data.status.toUpperCase()}`]);
             setStatus(data.status);
          }
          
          if (data.status === "finished" && data.result) {
            setVideoUrl(`${API_BASE}${data.result}`);
            setStatus("finished");
            setLogs((prev) => [...prev, "> SUCCESS: Video generation completed."]);
          } else if (data.status === "failed") {
            setStatus("failed");
            setLogs((prev) => [...prev, "> ERROR: Pipeline execution failed on node."]);
          }
        } catch (err) {
          console.error(err);
        }
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [jobId, status]);

  return (
    <div className="font-body-md text-body-md antialiased overflow-hidden flex h-screen bg-background text-on-background">
      {/* SideNavBar */}
      <nav className="w-[260px] h-screen fixed left-0 top-0 bg-surface-container dark:bg-surface-container text-primary dark:text-primary font-body-md text-body-md border-r border-outline-variant backdrop-blur-xl bg-opacity-50 shadow-lg flex flex-col p-base z-20">
        {/* Header */}
        <div className="px-4 py-6 border-b border-outline-variant/30 mb-4 flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-surface-container-highest flex items-center justify-center overflow-hidden border border-outline-variant">
            <img
              alt="User profile"
              className="w-full h-full object-cover"
              src="https://lh3.googleusercontent.com/aida-public/AB6AXuB57ScdKgV9Pg0OIZbJW1KBwLugzHx92D7yAeKFUWqUjvK_kw3XdPx9l2gx8lXkqwSggBtU013Oo1e2-ASoEEJpMPN126aX-Xo3H6BqCneXj7sVJXHCwSiwlbbU3RDnR12sNqHCaJTzbCs9nYG5qRkGKgql7uxnLWpa28ZI5FjMDKXHcGe9YZxr14W4Ns3Lr0dx8PbJ9WTJhuxvoQqoBb97ipaQVUcNYzcX34AEZY5FOVi4JVsDVwiajfEo_HP8xuGzfmElv--TD-w"
            />
          </div>
          <div>
            <h1 className="font-headline-md text-headline-md font-bold text-on-surface text-[16px] leading-tight">
              Cinematic Studio
            </h1>
            <p className="text-on-surface-variant text-sm">Pro Account</p>
          </div>
        </div>

        {/* Main Nav */}
        <div className="flex-1 overflow-y-auto">
          <ul className="space-y-1">
            <li>
              <a href="#" className="flex items-center gap-3 px-4 py-3 rounded-lg text-primary font-bold border-r-2 border-primary bg-primary/10 transition-all duration-200 hover:bg-surface-bright">
                <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>movie_filter</span>
                Studio
              </a>
            </li>
            <li>
              <a href="#" className="flex items-center gap-3 px-4 py-3 rounded-lg text-on-surface-variant hover:bg-surface-bright transition-all duration-200">
                <span className="material-symbols-outlined">video_library</span>
                Library
              </a>
            </li>
            <li>
              <a href="#" className="flex items-center gap-3 px-4 py-3 rounded-lg text-on-surface-variant hover:bg-surface-bright transition-all duration-200">
                <span className="material-symbols-outlined">hub</span>
                Network Nodes
              </a>
            </li>
            <li>
              <a href="#" className="flex items-center gap-3 px-4 py-3 rounded-lg text-on-surface-variant hover:bg-surface-bright transition-all duration-200">
                <span className="material-symbols-outlined">vpn_key</span>
                API Keys
              </a>
            </li>
            <li>
              <a href="#" className="flex items-center gap-3 px-4 py-3 rounded-lg text-on-surface-variant hover:bg-surface-bright transition-all duration-200">
                <span className="material-symbols-outlined">settings</span>
                Settings
              </a>
            </li>
          </ul>
        </div>

        {/* Footer Area & Credits Card */}
        <div className="mt-auto space-y-4 pt-4 border-t border-outline-variant/30">
          <div className="glass-panel p-4 rounded-xl flex flex-col gap-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-on-surface-variant">Credits</span>
              <span className="font-bold text-primary">1,240</span>
            </div>
            <div className="w-full h-1.5 bg-surface-container-highest rounded-full overflow-hidden">
              <div className="h-full bg-primary rounded-full w-[45%]"></div>
            </div>
            <button className="w-full py-2 bg-surface-container-highest hover:bg-surface-bright border border-outline-variant rounded-lg text-sm font-medium transition-colors">
              Top Up
            </button>
          </div>
          <ul className="space-y-1">
            <li>
              <a href="#" className="flex items-center gap-3 px-4 py-2 rounded-lg text-sm text-on-surface-variant hover:bg-surface-bright transition-all duration-200">
                <span className="material-symbols-outlined text-[18px]">help</span>
                Support
              </a>
            </li>
            <li>
              <a href="#" className="flex items-center gap-3 px-4 py-2 rounded-lg text-sm text-on-surface-variant hover:bg-surface-bright transition-all duration-200">
                <span className="material-symbols-outlined text-[18px]">description</span>
                Documentation
              </a>
            </li>
          </ul>
        </div>
      </nav>

      {/* TopAppBar */}
      <header className="fixed top-0 right-0 left-[260px] h-16 bg-surface-container-low border-b border-outline-variant backdrop-blur-md bg-opacity-30 z-10 flex justify-between items-center px-margin-desktop w-[calc(100%-260px)] font-technical-sm text-technical-sm text-primary">
        <div className="flex items-center gap-2 text-on-surface-variant">
          <a href="#" className="hover:text-primary transition-colors">Dashboard</a>
          <span className="material-symbols-outlined text-[16px] opacity-50">chevron_right</span>
          <span className="text-primary font-medium">Studio</span>
        </div>
        <div className="flex items-center gap-6">
          <div className="relative group hidden md:block">
            <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-on-surface-variant text-[18px]">search</span>
            <input
              type="text"
              className="bg-surface-container-highest border border-outline-variant rounded-lg py-1.5 pl-9 pr-12 text-sm text-on-surface focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary w-64 transition-all bg-opacity-50 backdrop-blur-sm"
              placeholder="Search prompts, models..."
            />
            <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
              <kbd className="font-technical-sm text-[10px] bg-surface-container px-1.5 py-0.5 rounded border border-outline-variant text-on-surface-variant">⌘</kbd>
              <kbd className="font-technical-sm text-[10px] bg-surface-container px-1.5 py-0.5 rounded border border-outline-variant text-on-surface-variant">K</kbd>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <button className="relative text-on-surface-variant hover:text-on-surface transition-colors">
              <span className="material-symbols-outlined">notifications</span>
              <span className="absolute top-0 right-0 w-2 h-2 bg-error rounded-full ring-2 ring-surface-container-low"></span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="ml-[260px] mt-16 flex h-[calc(100vh-64px)] w-[calc(100%-260px)] overflow-hidden">
        {/* Left Panel: Parameters */}
        <aside className="w-[400px] border-r border-outline-variant/50 bg-surface/50 backdrop-blur-sm flex flex-col h-full overflow-y-auto">
          <div className="p-6 space-y-8 pb-24">
            {/* Prompt Area */}
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <label className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">Cinematic Prompt</label>
                <span className="text-xs text-on-surface-variant opacity-70 font-technical-sm">{prompt.length}/1000</span>
              </div>
              <div className="relative">
                <textarea
                  className="w-full bg-surface-container border border-outline-variant rounded-xl p-4 text-on-surface font-technical-sm text-technical-sm resize-none focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary focus:neon-shadow-primary transition-all placeholder-on-surface-variant/50"
                  placeholder="Describe your scene in detail..."
                  rows={6}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                />
                <button 
                  type="button" 
                  className="absolute bottom-3 right-3 bg-surface-container-highest border border-outline-variant hover:border-secondary hover:text-secondary text-on-surface-variant px-3 py-1.5 rounded-lg flex items-center gap-2 text-xs font-medium transition-all group"
                  onClick={() => setPrompt(prompt + " cinematic lighting, highly detailed, 8k resolution, slow motion")}
                >
                  <span className="material-symbols-outlined text-[14px] text-secondary group-hover:animate-pulse">auto_awesome</span>
                  AI Enhance
                </button>
              </div>
            </div>

            {/* Model Selection */}
            <div className="space-y-3">
              <label className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">Engine</label>
              <div className="glass-panel rounded-xl overflow-hidden">
                <div className="p-4 flex justify-between items-center cursor-pointer hover:bg-white/[0.02] transition-colors border-b border-outline-variant/30">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded bg-primary/20 flex items-center justify-center text-primary">
                      <span className="material-symbols-outlined text-[18px]">memory</span>
                    </div>
                    <div>
                      <div className="text-sm font-medium text-on-surface">Wan2.1 T2V</div>
                      <div className="text-xs text-on-surface-variant">Text to Video • High Fidelity</div>
                    </div>
                  </div>
                  <span className="material-symbols-outlined text-on-surface-variant">expand_less</span>
                </div>
                <div className="p-2 bg-surface-container-lowest/50">
                  <div className="p-2 hover:bg-white/[0.05] rounded-lg cursor-pointer flex items-center gap-3 transition-colors">
                    <div className="w-6 h-6 rounded bg-surface-container-highest flex items-center justify-center">
                      <span className="material-symbols-outlined text-[14px] text-on-surface-variant">image</span>
                    </div>
                    <span className="text-sm text-on-surface-variant">SDXL Base (Image)</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Aspect Ratio */}
            <div className="space-y-3">
              <label className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">Aspect Ratio</label>
              <div className="flex gap-2 p-1 bg-surface-container rounded-lg border border-outline-variant/50">
                <button className="flex-1 py-2 bg-surface-container-highest text-primary border border-outline-variant/50 rounded flex items-center justify-center gap-2 font-technical-sm text-xs shadow-sm">
                  <span className="material-symbols-outlined text-[16px]">crop_16_9</span>
                  16:9
                </button>
                <button className="flex-1 py-2 text-on-surface-variant hover:text-on-surface hover:bg-surface-bright/50 rounded flex items-center justify-center gap-2 font-technical-sm text-xs transition-colors">
                  <span className="material-symbols-outlined text-[16px]">crop_9_16</span>
                  9:16
                </button>
                <button className="flex-1 py-2 text-on-surface-variant hover:text-on-surface hover:bg-surface-bright/50 rounded flex items-center justify-center gap-2 font-technical-sm text-xs transition-colors">
                  <span className="material-symbols-outlined text-[16px]">crop_square</span>
                  1:1
                </button>
              </div>
            </div>
          </div>

          {/* Sticky Render Button */}
          <div className="absolute bottom-0 left-0 w-full p-6 bg-gradient-to-t from-surface via-surface to-transparent border-t border-outline-variant/10">
            <button
              onClick={handleSubmit}
              disabled={status === "submitting" || status === "queued" || status === "started" || !prompt}
              className="w-full py-4 bg-gradient-to-b from-primary-container to-primary-fixed-variant text-on-primary-container font-bold rounded-xl shadow-[0_0_20px_rgba(77,142,255,0.3)] hover:shadow-[0_0_30px_rgba(77,142,255,0.5)] transform hover:scale-[1.02] disabled:opacity-50 disabled:scale-100 disabled:cursor-not-allowed transition-all duration-300 flex items-center justify-center gap-2"
            >
              {status === "queued" || status === "started" ? (
                <>
                  <span className="material-symbols-outlined animate-spin">refresh</span>
                  Processing Engine...
                </>
              ) : (
                <>
                  <span className="material-symbols-outlined">play_circle</span>
                  Render Video
                </>
              )}
            </button>
          </div>
        </aside>

        {/* Right Panel: Canvas & Terminal */}
        <section className="flex-1 flex flex-col p-6 gap-6 overflow-hidden relative">
          {/* Video Canvas */}
          <div className="flex-1 glass-panel rounded-2xl relative overflow-hidden flex flex-col items-center justify-center group bg-black">
            {!videoUrl ? (
              <>
                {/* Grid Background */}
                <div
                  className="absolute inset-0 opacity-10 pointer-events-none"
                  style={{
                    backgroundImage: "linear-gradient(#424754 1px, transparent 1px), linear-gradient(90deg, #424754 1px, transparent 1px)",
                    backgroundSize: "40px 40px"
                  }}
                />
                <div className="z-10 flex flex-col items-center gap-4 text-center">
                  <div className="w-16 h-16 rounded-2xl border-2 border-dashed border-outline-variant flex items-center justify-center text-outline-variant group-hover:border-primary/50 group-hover:text-primary/50 transition-colors">
                    {status === "queued" || status === "started" ? (
                      <span className="material-symbols-outlined text-3xl animate-pulse text-primary">hourglass_empty</span>
                    ) : (
                      <span className="material-symbols-outlined text-3xl">video_camera_front</span>
                    )}
                  </div>
                  <div>
                    <h3 className="text-on-surface font-medium mb-1">
                       {status === "queued" || status === "started" ? "Rendering in Progress..." : "Canvas Ready"}
                    </h3>
                    <p className="text-on-surface-variant text-sm max-w-[250px]">
                      {status === "queued" || status === "started" 
                        ? "Please wait while your video is generated on the Nosana network."
                        : "Enter a prompt and hit render to generate your cinematic sequence."}
                    </p>
                  </div>
                </div>
              </>
            ) : (
               <video 
                  src={videoUrl} 
                  controls 
                  autoPlay
                  loop
                  className="w-full h-full object-contain"
                />
            )}
            
            {/* Download Button overlay when video is ready */}
            {videoUrl && (
              <div className="absolute top-4 right-4 z-20">
                <a
                  href={videoUrl}
                  download
                  className="flex items-center gap-2 bg-surface-container-highest/80 backdrop-blur-md border border-outline-variant text-on-surface px-4 py-2 rounded-lg hover:bg-primary/20 hover:text-primary hover:border-primary transition-all shadow-lg"
                >
                  <span className="material-symbols-outlined text-[20px]">download</span>
                  <span className="text-sm font-medium">Download HD</span>
                </a>
              </div>
            )}
          </div>

          {/* Node Terminal */}
          <div className="h-48 bg-[#0a0a0a] border border-outline-variant/30 rounded-xl flex flex-col overflow-hidden font-technical-sm text-technical-sm shadow-inner relative">
            <div className="bg-surface-container-highest px-4 py-2 border-b border-outline-variant/30 flex justify-between items-center">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-[14px] text-on-surface-variant">terminal</span>
                <span className="text-xs text-on-surface-variant font-medium">Node Output Log</span>
              </div>
              <div className="flex gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-outline-variant/50"></div>
                <div className="w-2.5 h-2.5 rounded-full bg-outline-variant/50"></div>
                <div className="w-2.5 h-2.5 rounded-full bg-outline-variant/50"></div>
              </div>
            </div>
            <div className="p-4 flex-1 overflow-y-auto text-xs leading-relaxed opacity-80 flex flex-col gap-1">
               {logs.map((log, index) => {
                  let color = "text-[#8c909f]";
                  if (log.includes("ONLINE") || log.includes("SUCCESS")) color = "text-[#10b981]"; // green
                  if (log.includes("ERROR")) color = "text-error"; // red
                  if (log.includes("Status update")) color = "text-primary"; // blue
                  
                  return <div key={index} className={color}>{log}</div>
               })}
              <div className="text-[#8c909f] animate-pulse">_</div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
