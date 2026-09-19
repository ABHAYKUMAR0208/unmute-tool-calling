import { useEffect, useState } from "react";

export const useBackendServerUrl = () => {
  const [backendServerUrl, setBackendServerUrl] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window !== "undefined") {
      // If an explicit backend URL is set at build time, use it directly.
      const explicit = process.env.NEXT_PUBLIC_BACKEND_URL;
      if (explicit) {
        setBackendServerUrl(explicit.replace(/\/$/, ""));
        return;
      }

      // Docker mode: proxy through /api on the same host.
      const isInDocker = ["true", "1"].includes(
        process.env.NEXT_PUBLIC_IN_DOCKER?.toLowerCase() || ""
      );
      const prefix = isInDocker ? "/api" : "";
      const backendUrl = new URL("", window.location.href);
      if (!isInDocker) {
        backendUrl.port = "8000";
      }
      backendUrl.pathname = prefix;
      backendUrl.search = "";
      setBackendServerUrl(backendUrl.toString().replace(/\/$/, ""));
    }
  }, []);

  return backendServerUrl;
};
