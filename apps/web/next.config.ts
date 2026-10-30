import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  allowedDevOrigins: ["127.0.0.1"],
  async rewrites() {
    const apiBase = (process.env.VISIONWISE_API_INTERNAL_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
    return [{ source: "/vision-api/:path*", destination: `${apiBase}/:path*` }];
  },
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
