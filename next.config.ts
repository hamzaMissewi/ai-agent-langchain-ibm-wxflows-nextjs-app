import type {NextConfig} from "next";

// /** @type {import('next').NextConfig} */
const nextConfig: NextConfig = {
    eslint: {
        ignoreDuringBuilds: true
      }

    // /* config options here */
    // Hamza
    // experimental: {
    //     serverActions: true,
    // },
};

export default nextConfig;
