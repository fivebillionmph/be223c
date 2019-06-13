(function() {

var app = new Vue({
    el: "#app",
    delimiters: ["${", "}"],
    data: {
        image_file: null,
        image_data: null,
        image_point: null,
        result: null,
        error: null,
        loading: false,
        transform_size: globals.img_transform_size,
    },
    methods: {
        resetError: function() {
            this.error = null;
        },
        resetImageData: function() {
            this.image_file = null;
            this.image_data = null;
            this.image_point = null;
            this.result = null;
        },
        query: function() {
            var that = this;
            if(!this.submitable()) {
                return;
            }
            this.resetError();
            this.result = null;
            if(this.image_file === null) {
                this.error = "Please select a file first";
                return;
            }
            var form_data = new FormData();
            var req_data = {
                "point": this.image_point,
            };
            form_data.append("file", this.image_file);
            form_data.append("data", JSON.stringify(req_data));
            this.loading = true;
            axios.post("/api/query-image", form_data, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            }).then(function(response) {
                that.result = response.data;
                if(that.result.similar_images === undefined) {
                    that.result.similar_images = [];
                    console.log(Object.keys(that.result));
                    console.log(that.result["0"]);
                    console.log(that.result["1"]);
                    console.log(that.result["2"]);
                }
                that.result.similar_images.sort(function(a, b) {
                    if(b.similarity > a.similarity) {
                        return -1;
                    } else if(b.similarity < a.similarity) {
                        return 1;
                    }
                    return 0;
                });
            }, function(err) {
                that.error = err.data;
            }).finally(function() {
                that.loading = false;
            });
        },
        updateImageFile: function() {
            var that = this;
            this.resetError();
            this.resetImageData();
            if(this.$refs.file.files.length == 0) {
                this.image_file = null;
                return;
            }
            this.image_file = this.$refs.file.files[0];
            var reader = new FileReader();
            reader.onload = function(event) {
                that.image_data = btoa(event.target.result);
                that.addImagePoint();
            };
            reader.onerror = function(err) {
                that.error = "Could not read image file";
            };
            reader.readAsBinaryString(this.image_file);
        },
        addImagePoint: function(point) {
            var that = this;
            const RADIUS = 3;

            if(point !== undefined) {
                this.image_point = point;
            }

            var image_obj = new Image();
            image_obj.src = "data:image/jpeg;base64," + that.image_data;
            image_obj.onload = function() {
                var c = document.getElementById("canvas-image-region");
                var ctx = c.getContext("2d");
                ctx.drawImage(image_obj, 0, 0, that.transform_size[0], that.transform_size[1]);

                if(that.image_point) {
                    var pixel = ctx.createImageData(1,1);
                    pixel.data[0] = 255;
                    pixel.data[1] = 0;
                    pixel.data[2] = 0;
                    pixel.data[3] = 255;
                    for(var i = that.image_point.x - RADIUS; i <= that.image_point.x + RADIUS; i++) {
                        for(var j = that.image_point.y - RADIUS; j <= that.image_point.y + RADIUS; j++) {
                            ctx.putImageData(pixel, i, j);
                        }
                    }
                }
            };
        },
        submitable: function() {
            return this.image_file && this.image_data && this.image_point;
        },
        probString: function(prob) {
            return (Math.round(prob * 1000) / 1000).toString();
        },
    },
});

var canvas = document.getElementById("canvas-image-region");
var ctx = canvas.getContext("2d");
canvas.addEventListener("mouseup", function(e) {
    var point = getMousePos(canvas, e);
    app.addImagePoint(point);
});

function getMousePos(canvas, evt) {
    /* this function taken from: https://stackoverflow.com/a/17130415/3704042 */
    var rect = canvas.getBoundingClientRect();
    var scale_x = canvas.width / rect.width;
    var scale_y = canvas.height / rect.height;

    return {
        x: (evt.clientX - rect.left) * scale_x,
        y: (evt.clientY - rect.top) * scale_y,
    };
}

}());
