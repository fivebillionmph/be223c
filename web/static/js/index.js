(function() {

var app = new Vue({
    el: "#app",
    delimiters: ["${", "}"],
    data: {
        image_file: null,
        result: null,
        error: null,
    },
    methods: {
        resetError: function() {
            this.error = null;
        },
        query: function() {
            var that = this;
            this.resetError();
            this.result = null;
            if(this.image_file === null) {
                this.error = "Please select a file first";
                return;
            }
            var form_data = new FormData();
            form_data.append("file", this.image_file);
            axios.post("/api/query-image", form_data, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            }).then(function(response) {
                that.result = response.data;
                that.result.similar_images.sort(function(a, b) {
                    if(b.similarity > a.similarity) {
                        return 1;
                    } else if(b.similarity < a.similarity) {
                        return -1;
                    }
                    return 0;
                });
            }, function(err) {
                that.error = err.data;
            });
        },
        updateImageFile: function() {
            if(this.$refs.file.files.length == 0) {
                this.image_file = null;
                return;
            }
            this.image_file = this.$refs.file.files[0];
        },
    },
});

}());
