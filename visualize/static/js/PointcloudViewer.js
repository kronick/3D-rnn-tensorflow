
function PointcloudViewer() {
    this.settings = {
        draw_normals: false,
        draw_points: true,
    
        vertex_scale: 1,
        normal_scale: 0.3,
    }
    this.normals = [];

    this.initialize();
}

PointcloudViewer.prototype.loadPoints = function(points) {
    // Clean out the old stuff
    this.scene.remove(this.pointCloud);
    this.pointsGeo.dispose();
    for(var i=0; i<this.normalsGroup.children.length; i++) {
        this.normalsGroup.remove(this.normalsGroup.children[i]);
    }

    for(var i=0; i<this.normals.length; i++) {
        this.normals.dispose();
    }
    this.normals.length = 0

    this.pointsGeo = new THREE.Geometry();

    for(var i=0; i<points.length; i++) {
        var p = new THREE.Vector3(points[i][0], points[i][1], points[i][2]);
        p.multiplyScalar(this.settings.vertex_scale);
        this.pointsGeo.vertices.push(p);

        var c = new THREE.Color(1,1,1);
        this.pointsGeo.colors.push(c);

        // Create a normal line
        // var n = new THREE.Vector3(points[i][3], points[i][4], points[i][5]);
        // n.multiplyScalar(this.settings.normal_scale);
        // normalGeo = new THREE.Geometry();
        // normalGeo.vertices.push(p);
        // normalGeo.vertices.push(new THREE.Vector3().addVectors(p, n));
        // normalLine = new THREE.Line(normalGeo, this.normalMaterial);
        // this.normalsGroup.add(normalLine);
    }

    this.pointCloud = new THREE.Points(this.pointsGeo, this.pointsMaterial);
    this.scene.add(this.pointCloud);
    console.log("Point cloud loaded. Number of points: " + points.length);
}

PointcloudViewer.prototype.setColors = function(colors) {
    for(var i=0; i<colors.length; i++) {
        var c = new THREE.Color(colors[i][0], colors[i][1], colors[i][2]);
        this.pointsGeo.colors[i] = c;
    }

    this.pointsGeo.colorsNeedUpdate = true;
}

PointcloudViewer.prototype.setPointSize = function(size) {
    this.pointsMaterial.size = size;
}

PointcloudViewer.prototype.initialize = function() {
    //var model_file = "models/untitled.obj"
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 10000);
    this.renderer = new THREE.WebGLRenderer({antialias: true});
    this.renderer.setClearColor(0x444444);

    this.pointsGeo  = new THREE.Geometry();
    this.pointCloud = new THREE.Points(this.pointsGeo);
    this.scene.add(this.pointCloud);

    this.normalsGroup = new THREE.Object3D();
    this.scene.add(this.normalsGroup);

    this.renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(this.renderer.domElement);

    this.normalMaterial = new THREE.LineBasicMaterial({ color: 0xFFFFFF });
    this.normalMaterial.opacity = 0.3;
    this.normalMaterial.transparent = true;

    this.pointsMaterial = new THREE.PointsMaterial({ color: 0xFFFFFF, size: 0.5, vertexColors: THREE.VertexColors });

    // Set up camera and controls
    this.camera.position.z = 5;
    this.controls = new THREE.TrackballControls(this.camera, this.renderer.domElement);
    thisthis = this;
    this.controls.addEventListener('change', function() { thisthis.render(); });
    this.controls.rotateSpeed = 2.0;
    this.controls.zoomSpeed = 1.2;
    this.controls.panSpeed = 0.3;
    this.controls.noZoom = false;
    this.controls.noPan = false;
    this.controls.keys = [ 65, 83, 68 ];
    this.controls.staticMoving = false;
    this.controls.dynamicDampingFactor = 0.1;

    hemiLight = new THREE.HemisphereLight( 0xffffff, 0xddddff, 1.0 );
    hemiLight.color.setRGB( 1.0, 1.0, 1.0 );
    hemiLight.groundColor.setRGB( 0.5, 0.5, 1.0 );
    hemiLight.position.set( 25, 50, 0 );
    this.scene.add( hemiLight );

    this.animate();
}

PointcloudViewer.prototype.animate = function() {
    thisthis = this;
    requestAnimationFrame(function() { thisthis.animate(); });
    thisthis.controls.update();
    thisthis.render();
}

PointcloudViewer.prototype.render = function() {
    //requestAnimationFrame(render);
    this.renderer.render(this.scene, this.camera);

}

// UTILITIES
// -----------------------------------------------------

function randomInt(low, high) {
    return Math.floor(Math.random() * (high-low)) + low;
}
function randomItem(arr) {
    // Choose a random item from an array
    if(arr.length == 0) return;
    return arr[randomInt(0, arr.length)];
}