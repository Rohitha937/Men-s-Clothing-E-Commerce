document.addEventListener("click", async (event) => {
  const button = event.target.closest(".share-button");
  if (!button) return;

  const shareUrl = button.dataset.shareUrl;
  if (!shareUrl) return;

  try {
    if (navigator.share) {
      await navigator.share({
        title: "MR Product",
        url: shareUrl,
      });
    } else if (navigator.clipboard) {
      await navigator.clipboard.writeText(shareUrl);
      const originalLabel = button.textContent;
      button.textContent = "Link copied";
      setTimeout(() => {
        button.textContent = originalLabel;
      }, 1600);
    }
  } catch (error) {
    console.warn("Share action was cancelled or unavailable.", error);
  }
});
