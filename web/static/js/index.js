(function() {

var app = new Vue({
    el: "#app",
    delimiters: ["${", "}"],
    data: {
        image_file: null,
        results: null,
        error: null,
    },
    methods: {
        resetError: function() {
            this.error = null;
        },
        query: function() {
            var that = this;
            this.resetError();
            if(this.image_file === null) {
                this.error = "Please select a file first";
                return;
            }
            var form_data = new FormData();
            form_data.append("file", this.file);
            axios.post("/api/query-file", form_data, {
                headers: {
                    "Content-type": "multipart/form-data",
                },
            }).then(function(response) {
                that.results = response.data;
            }, function(err) {
                that.error = response.data;
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
