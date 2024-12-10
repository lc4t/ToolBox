// workers-site/index.ts
var worker = {
  async fetch(request) {
    try {
      const url = new URL(request.url);
      let path = url.pathname;
      if (path === "/" || path === "") {
        path = "/index.html";
      }
      const response = await fetch(request);
      if (response.ok) {
        const headers = new Headers(response.headers);
        headers.set("Cache-Control", "no-cache");
        return new Response(response.body, {
          status: response.status,
          headers
        });
      }
      if (!path.endsWith(".html")) {
        const indexResponse = await fetch(new URL("/index.html", url.origin));
        if (indexResponse.ok) {
          return new Response(indexResponse.body, {
            status: 200,
            headers: {
              "Content-Type": "text/html",
              "Cache-Control": "no-cache"
            }
          });
        }
      }
      return new Response("Not Found", {
        status: 404,
        headers: { "Content-Type": "text/plain" }
      });
    } catch (error) {
      console.error("Error serving:", error);
      return new Response("Internal Server Error", {
        status: 500,
        headers: { "Content-Type": "text/plain" }
      });
    }
  }
};
var workers_site_default = worker;
export {
  workers_site_default as default
};
