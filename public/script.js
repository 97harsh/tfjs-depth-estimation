/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1";

const STATUS = document.getElementById('status');
const VIDEO = document.getElementById('webcam');
const ENABLE_CAM_BUTTON = document.getElementById('enableCam');
const CLICK_PICTURE = document.getElementById('clickPicture');
const CANVAS = document.getElementById('canvas');

const STOP_DATA_GATHER = -1;


STATUS.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
// TODO: keep max of two greencrosses, if user clicks again on screen, remove both and start afresh with no green crosses
ENABLE_CAM_BUTTON.addEventListener('click', enableCam);
CLICK_PICTURE.addEventListener('click', clickPicture);




// crosscanvas is a transparent layer on top of my original image
const keyPoints = [];
const keyPointThreshold = 10; // Proximity threshold for removing a cross

// Create a new canvas element for the green crosses layer
const crossesCanvas = document.createElement('canvas');
crossesCanvas.width = CANVAS.width;
crossesCanvas.height = CANVAS.height;
console.log(CANVAS.width);
console.log(CANVAS.height);
crossesCanvas.style.position = 'absolute';
crossesCanvas.style.top = CANVAS.offsetTop + 'px';
crossesCanvas.style.left = CANVAS.offsetLeft + 'px';
CANVAS.parentNode.insertBefore(crossesCanvas, CANVAS.nextSibling);

crossesCanvas.addEventListener('click', markKeypoints);

async function predict(dataURL){
  try {
    // Pass the image data URL to the depth estimation model
    let mobilenet = await mobilenetPromise;
    const output = await mobilenet(dataURL);
    // console.log(output);
    // console.log(typeof(output));
    // output.depth.save('depth.png');
    return output;

  } catch (error) {
    console.error('Error during depth estimation:', error);
    throw error;
    }
}

async function clickPicture(){
  if (videoPlaying) {
    // Get the canvas element
    const ctx = CANVAS.getContext('2d');

    // Set the canvas dimensions to match the video stream
    canvas.width = VIDEO.videoWidth;
    canvas.height = VIDEO.videoHeight;

    // Draw the current frame from the video stream onto the canvas
    ctx.drawImage(VIDEO, 0, 0, canvas.width, canvas.height);

    const dataURL = canvas.toDataURL('image/png');
    predictOutput = await predict(dataURL);
    console.log(predictOutput);
  }
}

let mobilenetPromise = undefined;
let videoPlaying = false;
let predictOutput;

/**
 * Loads the MobileNet model and warms it up so ready for use.
 **/
async function loadDepthModel() {

    mobilenetPromise = pipeline('depth-estimation', 'Xenova/depth-anything-small-hf');

    STATUS.innerText = 'Depth Model loaded successfully!';

  }

  // Call the function immediately to start loading.
loadDepthModel();

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function enableCam() {
  if (hasGetUserMedia()) {
    // getUsermedia parameters.
    const constraints = {
      video: true,
      width: 640, 
      height: 480 
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
      VIDEO.srcObject = stream;
      VIDEO.addEventListener('loadeddata', function() {
        videoPlaying = true;
        ENABLE_CAM_BUTTON.classList.add('removed');
      });
    });
  } else {
    console.warn('getUserMedia() is not supported by your browser');
  }
}



function markKeypoints(event) {
  console.log("click");
  // Get the click coordinates relative to the canvas
  const rect = crossesCanvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  // Check if a green cross exists at the clicked location
  const existingCrossIndex = keyPoints.findIndex(cross =>
    Math.abs(cross.x - x) <= keyPointThreshold && Math.abs(cross.y - y) <= keyPointThreshold
  );

  if (existingCrossIndex !== -1) {
    // Remove the existing green cross and its coordinates from the array
    keyPoints.splice(existingCrossIndex, 1);
    // KeyPointDepth.splice(existingCrossIndex, 1);
    redrawCrossesCanvas();
  } else {
    if (keyPoints.length === 2) {
      // If two green crosses already exist, remove all crosses
      keyPoints.length = 0;
      // KeyPointDepth.length = 0;
      redrawCrossesCanvas();
    } else {
      // Add a new green cross at the clicked location
      keyPoints.push({ x, y });
      drawGreenCross(x, y);
    }
  }
}

function drawGreenCross(x, y) {
  const ctx = crossesCanvas.getContext('2d');
  
  // Draw the green cross
  ctx.strokeStyle = 'green';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x - 5, y);
  ctx.lineTo(x + 5, y);
  ctx.moveTo(x, y - 5);
  ctx.lineTo(x, y + 5);
  ctx.stroke();
  
  // Get the depth value from predictOutput at the cross position
  const depthValue = predictOutput.predicted_depth[x][y].data[0].toFixed(2);
  // KeyPointDepth.push(depthValue);
  // // Write the depth value on top of the cross
  if (depthValue) {
    ctx.font = '12px Arial';
    ctx.fillStyle = 'green';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(depthValue, x, y - 8);
  }
}

function redrawCrossesCanvas() {
  const ctx = crossesCanvas.getContext('2d');
  ctx.clearRect(0, 0, crossesCanvas.width, crossesCanvas.height);
  keyPoints.forEach(cross => drawGreenCross(cross.x, cross.y));
}

