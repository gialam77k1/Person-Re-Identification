import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  experimental: {
    // The compiler API is deterministic on the local/self-hosted runner;
    // CLI capture can return an empty --showConfig stream on this host.
    useTypeScriptCli: false,
  },
};

export default nextConfig;
