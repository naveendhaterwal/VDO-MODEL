import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Nosana Cinematic Studio",
  description: "Local AI Cinematic Video Generation",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-gray-900 text-white min-h-screen">
        <main className="container mx-auto p-8">
          <header className="mb-8 border-b border-gray-800 pb-4">
            <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
              Nosana Cinematic Studio
            </h1>
            <p className="text-gray-400 mt-2">ViMax + Wan2.1 Local Engine</p>
          </header>
          {children}
        </main>
      </body>
    </html>
  );
}
