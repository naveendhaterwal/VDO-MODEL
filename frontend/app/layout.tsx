import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Cinematic Studio",
  description: "Decentralized AI Cinematic Video Generation",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-background text-on-background min-h-screen m-0 p-0 overflow-hidden">
        {children}
      </body>
    </html>
  );
}
