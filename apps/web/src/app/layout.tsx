import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vision Wise | Live Assistance",
  description:
    "Voice-first visual assistance using an ESP32-CAM stream and local object detection.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
