const video = document.querySelector("#videoElement");

if (navigator.mediaDevices.getUserMedia) {
	navigator.mediaDevices.getUserMedia({ video: true })
        .then( (stream) => { 
          video.srcObject = stream;
    	})
  		.catch(function (error) {
          console.log("Something went wrong!");
          console.log(error);
          return;
        });
};
