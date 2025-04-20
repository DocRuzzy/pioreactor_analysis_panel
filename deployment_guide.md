# Deploying to Hugging Face Spaces

This guide walks you through deploying your Pioreactor Dilution Rate Analysis Tool to Hugging Face Spaces, which provides free hosting for academic projects.

## Prerequisites

- A Hugging Face account (sign up at https://huggingface.co/join)
- Your code in a GitHub repository

## Deployment Steps

### 1. Prepare Your Files

Make sure your repository has these essential files:
- `pioreactor_panel.py` (the main code for the Panel application)
- `app.py` (the entry point for Hugging Face Spaces)
- `requirements.txt` (with Panel 1.2.0+ specified)
- `README.md` (documentation for the repository)

### 2. Create a New Space

1. Log in to Hugging Face
2. Go to your profile and click "New Space"
3. Choose a name for your space (e.g., "pioreactor-analysis")
4. For Space SDK, select "Gradio" or "Streamlit" (both work with Panel)
5. Choose "Public" visibility for academic sharing
6. Click "Create Space"

### 3. Connect Your GitHub Repository

1. In your new Space, go to the "Settings" tab
2. Under "Repository", select "From existing repository"
3. Enter your GitHub repository URL
4. Select the branch to deploy (usually "main")
5. Click "Save"

### 4. Wait for Deployment

Hugging Face will automatically detect your Panel application and deploy it.
This process usually takes 2-5 minutes.

### 5. Configure for Academic Citation

1. Go to your Space's "Settings" tab
2. Under "Metadata", add relevant tags such as "bioreactor", "data-visualization", etc.
3. Add a detailed description explaining the project's academic purpose
4. For academic citation, you can also enable DOI integration

### 6. Share Your Space

Once deployed, you'll get a URL like:
```
https://huggingface.co/spaces/your-username/pioreactor-analysis
```

You can share this URL in your journal articles, papers, or academic publications.

## Troubleshooting

If your deployment fails, check these common issues:

1. **Dependencies**: Make sure your `requirements.txt` includes all needed packages
2. **Panel Version**: Verify you're using Panel 1.2.0+ features only
3. **File Structure**: Ensure `app.py` is in the root directory
4. **Logs**: Check the build logs for specific errors

## Getting a DOI for Academic Citation

For formal academic citations, you can get a DOI for your Space:

1. Go to Zenodo or Figshare and create an account
2. Upload a ZIP of your code or link your GitHub repository
3. Publish a new version to get a DOI
4. Add this DOI to your README.md and Space description
