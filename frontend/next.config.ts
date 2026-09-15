import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Frontend is static-friendly; API calls go to external FastAPI backend
  reactStrictMode: true,
};

export default nextConfig;
