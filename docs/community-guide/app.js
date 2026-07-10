window.addEventListener("DOMContentLoaded", () => {
  import("./lib/app-main.mjs")
    .then(({ initApp }) => {
      initApp();
    })
    .catch((error) => {
      console.error("Failed to initialize the WRF guide UI.", error);
    });
});
