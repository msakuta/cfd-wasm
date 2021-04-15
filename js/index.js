import("../pkg/index.js").catch(console.error).then(run);

async function run(module) {
  const { turing } = module;

  const canvasScale = 2.;
  const canvas = document.getElementById('canvas');
  const canvasSize = canvas.getBoundingClientRect();
  canvas.style.width = canvasSize.width * canvasScale + "px";
  canvas.style.height = canvasSize.height * canvasScale + "px";

  let params = {
    deltaTime: 1.0,
    skipFrames: 1,
    visc: 0.01,
    diff: 0.,
    density: 50.0,
    decay: 0.01,
    mouseFlowSpeed: 0.75,
    boundaryFlowSpeed: 0.02,
    mouseFlow: true,
    obstacle: false,
    dyeFromObstacle: true,
    boundaryX: "Fixed",
    boundaryY: "Fixed",
    diffIter: 4,
    projIter: 10,
    mousePos: undefined,
  };
  const ctx = canvas.getContext('2d');

  const animateCheckbox = document.getElementById("animate");
  const sliderUpdater = [];
  function sliderInit(sliderId, labelId, writer, logarithmic=false){
    const slider = document.getElementById(sliderId);
    const label = document.getElementById(labelId);
    label.innerHTML = slider.value;

    const update = (_event) => {
      let value;
      if(logarithmic){
        const minp = parseFloat(slider.getAttribute("min"));
        const maxp = parseFloat(slider.getAttribute("max"));

        // The result should be between 100 an 10000000
        const minv = Math.log(minp);
        const maxv = Math.log(maxp);

        // calculate adjustment factor
        const scale = (maxv-minv) / (maxp-minp);

        value = Math.exp(minv + scale*(parseFloat(slider.value) - minp));
        label.innerHTML = value.toFixed(8);
      }
      else{
        value = parseFloat(slider.value);
        label.innerHTML = value;
      }
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
  const deltaTimeSlider = sliderInit("deltaTime", "deltaTimeLabel", value => params.deltaTime = value);
  const skipFramesSlider = sliderInit("skipFrames", "skipFramesLabel", value => params.skipFrames = value);
  const fSlider = sliderInit("visc", "viscLabel", value => params.visc = value, true);
  const diffSlider = sliderInit("diff", "diffLabel", value => params.diff = value, true);
  const densitySlider = sliderInit("density", "densityLabel", value => params.density = value);
  const decaySlider = sliderInit("decay", "decayLabel", value => params.decay = value);
  const mouseFlowSpeedSlider = sliderInit("mouseFlowSpeed", "mouseFlowSpeedLabel", value => params.mouseFlowSpeed = value);
  const boundaryFlowSpeedSlider = sliderInit("boundaryFlowSpeed", "boundaryFlowSpeedLabel", value => params.boundaryFlowSpeed = value);
  const mouseFlowCheck = checkboxInit("mouseFlow", value => params.mouseFlow = value);
  const obstacleCheck = checkboxInit("obstacle", value => params.obstacle = value);
  const dyeFromObstacleCheck = checkboxInit("dyeFromObstacle", value => params.dyeFromObstacle = value);
  const wrapXCheck = radioButtonInit("wrapX", value => params.boundaryX = value);
  const fixedXCheck = radioButtonInit("fixedX", value => params.boundaryX = value);
  const flowXCheck = radioButtonInit("flowX", value => params.boundaryX = value);
  const wrapYCheck = radioButtonInit("wrapY", value => params.boundaryY = value);
  const fixedYCheck = radioButtonInit("fixedY", value => params.boundaryY = value);
  const flowYCheck = radioButtonInit("flowY", value => params.boundaryY = value);
  const diffIterSlider = sliderInit("diffIter", "diffIterLabel", value => params.diffIter = value);
  const projIterSlider = sliderInit("projIter", "projIterLabel", value => params.projIter = value);
  let resetParticles = false;

  const buttonResetParticles = document.getElementById("buttonResetParticles");
  buttonResetParticles.addEventListener("click", (event) => {
    resetParticles = true;
  })

  canvas.addEventListener("mousemove", (event) => {
      params.mousePos = [event.offsetX / canvasScale, event.offsetY / canvasScale];
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
                    ...params,
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
