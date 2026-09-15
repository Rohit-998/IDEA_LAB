import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ViolenceGuard — Motion-Gated Cascade Detection",
  description:
    "Real-time violence detection dashboard comparing always-on inference vs. the motion-energy cascade (cheap first check). Built on the RWF-2000 Flow-Gated Network.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
