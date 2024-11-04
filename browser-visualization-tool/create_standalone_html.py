import sys
import os
import json

def generate_html(image_path, json_path, output_html_name):
    # Read image data (if you wish to embed the image, you can uncomment the related code)
    image_name = os.path.basename(image_path)
    image_src = image_name  # Using relative path to reference the image

    # Read JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Convert JSON data to a JSON string for embedding
    json_data_str = json.dumps(json_data)

    # HTML template with placeholders
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Image Visualization Tool - {image_name}</title>
        <style>
            canvas {{
                border: 1px solid black;
                cursor: crosshair;
            }}
            #controls {{
                margin-bottom: 10px;
            }}
            .imageInfo {{
                margin-bottom: 5px;
            }}
            #textInfo {{
                margin-top: 10px;
                position: absolute;
                background-color: white;
                border: 1px solid black;
                padding: 5px;
                display: none;
            }}
        </style>
    </head>
    <body>
        <div id="controls">
            <!-- Controls -->
            <button id="zoomInButton">Zoom In</button>
            <button id="zoomOutButton">Zoom Out</button>
            <input type="text" id="searchInput" placeholder="Search text">
            <button id="searchButton">Search</button>
        </div>
        <canvas id="canvas"></canvas>
        <div id="textInfo"></div>

        <script>
            let scaleFactor = 1.0;
            const scaleStep = 0.1;
            const imageSrc = '{image_src}';

            // Embed the JSON data directly
            let jsonData = {json_data_str};

            function loadImageAndCoords() {{
                const imageElement = new Image();
                imageElement.src = imageSrc;
                imageElement.onload = function() {{
                    const canvas = document.getElementById('canvas');
                    canvas.width = imageElement.width * scaleFactor;
                    canvas.height = imageElement.height * scaleFactor;
                    const ctx = canvas.getContext('2d');

                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);

                    drawTextRegions(ctx, jsonData);
                }};
            }}

            function drawTextRegions(ctx, data) {{
                ctx.strokeStyle = 'red';
                data.forEach(item => {{
                    drawPolygon(ctx, item.coords);
                }});
            }}

            function drawPolygon(ctx, coords) {{
                ctx.beginPath();
                const points = coords.split(' ').map(point => point.split(',').map(Number));
                ctx.moveTo(points[0][0] * scaleFactor, points[0][1] * scaleFactor);
                for (let i = 1; i < points.length; i++) {{
                    ctx.lineTo(points[i][0] * scaleFactor, points[i][1] * scaleFactor);
                }}
                ctx.closePath();
                ctx.stroke();
            }}

            // Mouse event handling
            const canvas = document.getElementById('canvas');
            canvas.addEventListener('mousemove', function(event) {{
                const rect = canvas.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const y = event.clientY - rect.top;
                let foundHover = false;

                for (let i = 0; i < jsonData.length; i++) {{
                    const item = jsonData[i];
                    const points = item.coords.split(' ').map(point => point.split(',').map(Number));
                    const ctx = canvas.getContext('2d');

                    ctx.beginPath();
                    ctx.moveTo(points[0][0] * scaleFactor, points[0][1] * scaleFactor);
                    for (let j = 1; j < points.length; j++) {{
                        ctx.lineTo(points[j][0] * scaleFactor, points[j][1] * scaleFactor);
                    }}
                    ctx.closePath();

                    if (ctx.isPointInPath(x, y)) {{
                        foundHover = true;
                        showTextInfo(event.clientX, event.clientY, item.text);
                        break;
                    }}
                }}

                if (!foundHover) {{
                    clearTextInfo();
                }}
            }});

            function showTextInfo(x, y, text) {{
                const textInfo = document.getElementById('textInfo');
                textInfo.style.left = `${{x + 15}}px`;
                textInfo.style.top = `${{y + 15}}px`;
                textInfo.textContent = text;
                textInfo.style.display = 'block';
            }}

            function clearTextInfo() {{
                const textInfo = document.getElementById('textInfo');
                textInfo.style.display = 'none';
            }}

            // Zoom controls
            document.getElementById('zoomInButton').addEventListener('click', function() {{
                scaleFactor += scaleStep;
                loadImageAndCoords();
            }});

            document.getElementById('zoomOutButton').addEventListener('click', function() {{
                if (scaleFactor > scaleStep) {{
                    scaleFactor -= scaleStep;
                    loadImageAndCoords();
                }}
            }});

            // Search functionality
            document.getElementById('searchButton').addEventListener('click', function() {{
                const searchText = document.getElementById('searchInput').value.toLowerCase();
                const found = jsonData.some(item => item.text && item.text.toLowerCase().includes(searchText));
                if (found) {{
                    alert('Text found in this image.');
                }} else {{
                    alert('Text not found in this image.');
                }}
            }});

            // Initialize the image
            loadImageAndCoords();
        </script>
    </body>
    </html>
    """

    # Write the HTML content to the output file
    with open(output_html_name, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"Generated {output_html_name}")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python generate_html.py <image_path> <json_path> <output_html_name>")
    else:
        image_path = sys.argv[1]
        json_path = sys.argv[2]
        output_html_name = sys.argv[3]
        generate_html(image_path, json_path, output_html_name)
