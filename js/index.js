import("../pkg/index.js").catch(console.error).then(run);

async function run(module) {
  const { turing } = module;

  const canvasScale = 2.;
  const canvas = document.getElementById('canvas');
  const canvasSize = canvas.getBoundingClientRect();
  canvas.style.width = canvasSize.width * canvasScale + "px";
  canvas.style.height = canvasSize.height * canvasScale + "px";

  var deltaTime = 1.0;
  var skipFrames = 1;
  var visc = 0.01;
  var diff = 0.;
  var density = 50.0;
  var decay = 0.01;
  var rv = 0.75;
  var mouseFlow = true;
  var obstacle = false;
  var dyeFromObstacle = true;
  var boundaryX = "Fixed";
  var boundaryY = "Fixed";
  var diffIter = 4;
  var projIter = 10;
  var mousePos;

  const ctx = canvas.getContext('2d');

  const animateCheckbox = document.getElementById("animate");
  const sliderUpdater = [];
  function sliderInit(sliderId, labelId, writer){
    const slider = document.getElementById(sliderId);
    const label = document.getElementById(labelId);
    label.innerHTML = slider.value;

    const update = (_event) => {
        label.innerHTML = slider.value;
        writer(parseFloat(slider.value));
    }
    slider.addEventListener("input", update);
    sliderUpdater.push(update);
    return slider;
  }
  function checkboxInit(checkboxId, writer){
    const checkbox = document.getElementById(checkboxId);
    const update = (_event) => {
      writer(checkbox.checked);
    }
    checkbox.addEventListener("click", update);
    return checkbox;
  }
  function radioButtonInit(radioButtonId, writer){
    const radioButton = document.getElementById(radioButtonId);
    const update = (_event) => {
      if(radioButton.checked)
        writer(radioButton.value);
    }
    radioButton.addEventListener("click", update);
    return radioButton;
  }
  const deltaTimeSlider = sliderInit("deltaTime", "deltaTimeLabel", value => deltaTime = value);
  const skipFramesSlider = sliderInit("skipFrames", "skipFramesLabel", value => skipFrames = value);
  const fSlider = sliderInit("visc", "viscLabel", value => visc = value);
  const diffSlider = sliderInit("diff", "diffLabel", value => diff = value);
  const densitySlider = sliderInit("density", "densityLabel", value => density = value);
  const decaySlider = sliderInit("decay", "decayLabel", value => decay = value);
  const rvSlider = sliderInit("velo", "veloLabel", value => rv = value);
  const mouseFlowCheck = checkboxInit("mouseFlow", value => mouseFlow = value);
  const obstacleCheck = checkboxInit("obstacle", value => obstacle = value);
  const dyeFromObstacleCheck = checkboxInit("dyeFromObstacle", value => dyeFromObstacle = value);
  const wrapXCheck = radioButtonInit("wrapX", value => boundaryX = value);
  const fixedXCheck = radioButtonInit("fixedX", value => boundaryX = value);
  const flowXCheck = radioButtonInit("flowX", value => boundaryX = value);
  const wrapYCheck = radioButtonInit("wrapY", value => boundaryY = value);
  const fixedYCheck = radioButtonInit("fixedY", value => boundaryY = value);
  const flowYCheck = radioButtonInit("flowY", value => boundaryY = value);
  const diffIterSlider = sliderInit("diffIter", "diffIterLabel", value => diffIter = value);
  const projIterSlider = sliderInit("projIter", "projIterLabel", value => projIter = value);
  let resetParticles = false;

  const buttonResetParticles = document.getElementById("buttonResetParticles");
  buttonResetParticles.addEventListener("click", (event) => {
    resetParticles = true;
  })

  canvas.addEventListener("mousemove", (event) => {
      mousePos = [event.offsetX / canvasScale, event.offsetY / canvasScale];
  })

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
                const ret = {
                    terminate: !animateCheckbox.checked,
                    mousePos,
                    deltaTime,
                    skipFrames,
                    visc,
                    diff,
                    density,
                    decay,
                    rv,
                    mouseFlow,
                    obstacle,
                    dyeFromObstacle,
                    boundaryX,
                    boundaryY,
                    diffIter,
                    projIter,
                    resetParticles,
                };
                resetParticles = false;
                return ret;
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

  startAnimation();
}
