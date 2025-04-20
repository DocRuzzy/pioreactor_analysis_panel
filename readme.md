# Pioreactor Dilution Rate Analysis Tool

An interactive web application for analyzing bioreactor dilution rates from Pioreactor data. This tool is designed for academic research and can be used to analyze the performance of continuous culture bioreactors.

## Features

- Interactive visualization of dilution rates over time
- OD (Optical Density) tracking with target vs actual comparison
- Analysis of inter-dosing periods
- Statistical breakdown by target OD regions
- Bookmark system for marking important points
- CSV data upload capability

## Deployment

This application is deployed on [Hugging Face Spaces](https://huggingface.co/spaces) and can be accessed at: [insert your Spaces URL here]

## Local Development

To run this application locally:

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```
4. Open your browser to `http://localhost:5006`

## Usage Instructions

1. Upload your Pioreactor events CSV file using the file upload widget
2. Adjust reactor volume and moving average window settings as needed
3. Click "Update" to refresh the analysis
4. Click on points in the graphs to add bookmarks
5. View statistics organized by OD target regions

## Citation

If you use this tool in your research, please cite:

```
[Your Name]. (2025). Pioreactor Dilution Rate Analysis Tool. 
Hugging Face Spaces. https://huggingface.co/spaces/[your-username]/[space-name]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
