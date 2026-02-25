import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";

export const metadata: Metadata = {
  title: "UmaEdge — AI Horse Racing Intelligence",
  description: "AI-powered Japanese horse racing intelligence platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <Nav />
        <main>{children}</main>
      </body>
    </html>
  );
}
