import("../pkg/index.js").catch(console.error).then(run);

async function run(module) {
  const { cfd_canvas, cfd_webgl } = module;

  const canvasContainer = document.getElementById("canvasContainer");
  let canvasScale = 1.;
  let pixelScale = 4.;

  let canvas;
  let canvasSize;

  let useWebGL = true;
  let [width, height] = [200, 200];
  let pendingRestart = false;

  function setUseWebGL(value){
    useWebGL = value;
    [pixelScale, canvasScale] = useWebGL ? [1., 4] : [4., 1.];
    resizeCanvas();
    pendingRestart = true;
  }

  function resizeCanvas(){
    if(canvas)
      canvasContainer.removeChild(canvas);
    canvas = document.createElement("canvas");
    canvasContainer.appendChild(canvas);
    canvas.setAttribute("width", width * canvasScale);
    canvas.setAttribute("height", height * canvasScale);
    canvas.style.width = width * canvasScale * pixelScale + "px";
    canvas.style.height = height * canvasScale * pixelScale + "px";
    canvasSize = canvas.getBoundingClientRect();
    canvas.addEventListener("mousemove", (event) => {
      params.mousePos = [event.offsetX / canvasScale / pixelScale, event.offsetY / canvasScale / pixelScale];
    });
  }

  setUseWebGL(true);

  let params = {
    deltaTime: 1.0,
    skipFrames: 1,
    visc: 1e-7,
    diff: 0.,
    density: 50.0,
    decay: 0.01,
    mouseFlowSpeed: 0.75,
    boundaryFlowSpeed: 0.02,
    temperature: false,
    halfHeatSource: false,
    heatExchangeRate: 0.1,
    heatBuoyancy: 0.05,
    mouseFlow: true,
    showVelocity: true,
    showVelocityField: false,
    obstacle: false,
    dyeFromObstacle: true,
    particlesLabel: true,
    particleTrails: 0,
    boundaryX: "Fixed",
    boundaryY: "Fixed",
    diffIter: 4,
    projIter: 10,
    mousePos: undefined,
  };

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
  const useWebGLCheck = checkboxInit("useWebGL", setUseWebGL);
  const deltaTimeSlider = sliderInit("deltaTime", "deltaTimeLabel", value => params.deltaTime = value);
  const skipFramesSlider = sliderInit("skipFrames", "skipFramesLabel", value => params.skipFrames = value);
  const fSlider = sliderInit("visc", "viscLabel", value => params.visc = value, true);
  const diffSlider = sliderInit("diff", "diffLabel", value => params.diff = value, true);
  const densitySlider = sliderInit("density", "densityLabel", value => params.density = value);
  const decaySlider = sliderInit("decay", "decayLabel", value => params.decay = value);
  const mouseFlowSpeedSlider = sliderInit("mouseFlowSpeed", "mouseFlowSpeedLabel", value => params.mouseFlowSpeed = value);
  const boundaryFlowSpeedSlider = sliderInit("boundaryFlowSpeed", "boundaryFlowSpeedLabel", value => params.boundaryFlowSpeed = value);
  const mouseFlowCheck = checkboxInit("mouseFlow", value => params.mouseFlow = value);
  const showVelocityCheck = checkboxInit("showVelocity", value => params.showVelocity = value);
  const showVelocityFieldCheck = checkboxInit("showVelocityField", value => params.showVelocityField = value);
  const obstacleCheck = checkboxInit("obstacle", value => params.obstacle = value);
  const dyeFromObstacleCheck = checkboxInit("dyeFromObstacle", value => params.dyeFromObstacle = value);
  const temperatureCheck = checkboxInit("temperature", value => params.temperature = value);
  const halfHeatSourceCheck = checkboxInit("halfHeatSource", value => params.halfHeatSource = value);
  const heatExchangeRateSlider = sliderInit("heatExchangeRate", "heatExchangeRateLabel", value => params.heatExchangeRate = value);
  const heatBuoyancySlider = sliderInit("heatBuoyancy", "heatBuoyancyLabel", value => params.heatBuoyancy = value, true);
  const particlesCheck = checkboxInit("particles", value => params.particles = value);
  const particleTrailsSlider = sliderInit("particleTrails", "particleTrailsLabel", value => params.particleTrails = value);
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

  const label = document.getElementById('label');

  function startAnimation(){
    console.time('Rendering in Rust')
    try{
      const callback = (particles) => {
        // ctx.lineWidth = 1.;
        // ctx.strokeStyle = "#ffffff";
        // ctx.beginPath();
        // for(let i = 0; i < particles.length / 2; i++){
        //   ctx.moveTo(particles[i * 2], particles[i * 2 + 1]);
        //   ctx.lineTo(particles[i * 2], particles[i * 2 + 1] + 1);
        // }
        // ctx.stroke();
        // ctx.putImageData(data, 0, 0);
          label.innerHTML = particles.buffer;
          const animateCheckbox = document.getElementById("animate");
          const ret = {
              terminate: !animateCheckbox.checked || pendingRestart,
              ...params,
              resetParticles,
          };
          resetParticles = false;
          if(pendingRestart){
            setTimeout(startAnimation, 0);
            pendingRestart = false;
          }
          return ret;
      };

      if(useWebGL){
        const ctx = canvas.getContext('webgl');
        cfd_webgl(width, height, ctx, callback);
      }
      else{
        const ctx = canvas.getContext('2d');
        cfd_canvas(width, height, ctx, callback);
      }
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
