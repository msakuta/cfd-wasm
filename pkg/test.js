import init, { render_func } from './ray_rust_wasm.js'
import { deserialize_string, turing } from './ray_rust_wasm.js';

async function run() {
  await init()

  const canvas = document.getElementById('canvas');
  const canvasSize = canvas.getBoundingClientRect();

  var x = 0;
  var y = -150.;
  var z = -300.;
  var yaw = -90.;
  var pitch = -90.;

  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(canvasSize.width, canvasSize.height);

  const yaml = await fetch("./out.yaml");

  if (!yaml.ok || yaml.status !== 200)
    return;

  const yamlText = await yaml.text();

  const animateCheckbox = document.getElementById("animate");

  function renderCanvas(){
    if(animateCheckbox.checked)
        return;
    console.time('Rendering in Rust')
    try{
      const buf = render_func(ctx, canvasSize.width, canvasSize.height, [x, y, z],
        [0., yaw, pitch].map(deg => deg * Math.PI / 180));
    }
    catch(e){
      console.log("Rendering error: " + e);
    }
    console.timeEnd('Rendering in Rust')
  }

  var label = document.getElementById('label');

  function startAnimation(){
    console.time('Rendering in Rust')
    try{
    //   deserialize_string(yamlText, canvasSize.width, canvasSize.height,
        turing(canvasSize.width, canvasSize.height,
            (data, average) => {
                ctx.putImageData(data, 0, 0);
                label.innerHTML = average;
                const animateCheckbox = document.getElementById("animate");
                return !animateCheckbox.checked;
            });
    }
    catch(e){
        console.log("Rendering error: " + e);
    }
    console.timeEnd('Rendering in Rust')
  }

  animateCheckbox.onclick = (_event) => {
      if(animateCheckbox.checked)
        startAnimation();
  }

  var buttonStates = {
      w: false,
      s: false,
      a: false,
      d: false,
      q: false,
      z: false,
      ArrowRight: false,
      ArrowLeft: false,
      ArrowUp: false,
      ArrowDown: false,
  };
  function updatePos(){
      renderCanvas();
      label.innerHTML = `x=${x}<br>y=${y}<br>z=${z}<br>yaw=${yaw}<br>pitch=${pitch}`;
  }
  function tryUpdate(){
      var ok = false;
      if(buttonStates.a){
          x += 10 * Math.sin(yaw * Math.PI / 180);
          z += 10 * Math.cos(yaw * Math.PI / 180);
          ok = true;
      }
      if(buttonStates.d){
          x -= 10 * Math.sin(yaw * Math.PI / 180);
          z -= 10 * Math.cos(yaw * Math.PI / 180);
          ok = true;
      }
      if(buttonStates.w){
          x += 10 * Math.cos(yaw * Math.PI / 180);
          z -= 10 * Math.sin(yaw * Math.PI / 180);
          ok = true;
      }
      if(buttonStates.s){
          x -= 10 * Math.cos(yaw * Math.PI / 180);
          z += 10 * Math.sin(yaw * Math.PI / 180);
          ok = true;
      }
      if(buttonStates.q){
          y += 10;
          ok = true;
      }
      if(buttonStates.z){
          y -= 10;
          ok = true;
      }
      if(buttonStates.ArrowRight){
          yaw += 5;
          ok = true;
      }
      if(buttonStates.ArrowLeft){
          yaw -= 5;
          ok = true;
      }
      if(buttonStates.ArrowUp){
          pitch -= 5;
          ok = true;
      }
      if(buttonStates.ArrowDown){
          pitch += 5;
          ok = true;
      }
      if(ok){
          updatePos();
          return true;
      }
      return false;
  }
  updatePos();
  window.onkeydown = function(event){
      if(event.key in buttonStates){
          if(!buttonStates[event.key]){
              console.log(`onkeydown x: ${x}, y: ${y}`)
              buttonStates[event.key] = true;
              tryUpdate();
          }
          event.preventDefault();
      }
  }
  window.onkeyup = function(event){
      if(event.key in buttonStates){
          buttonStates[event.key] = false;
          event.preventDefault();
      }
  }
}

run()