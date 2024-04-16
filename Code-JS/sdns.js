// SDNS 


'use strict';


function shuffle(a,i,r,t){i=a.length;while(i)r=Math.random()*i--|0,t=a[i],a[i]=a[r],a[r]=t}


class SChunk{
	constructor(p1,p2){
		if(p1){
			this.parent1=p1;
			this.parent2=p2;
			p1.subchunks.push(this);
			p2.subchunks.push(this);
		}
		this.f=0;
		this.outputs=new Proxy({},{get:(o,k)=>o[k]||0});
		this.subchunks=[];
	}
	allowActivation(currentTic){
		return this.parent1.lastActivationTime==currentTic &&
			this.parent2.lastActivationTime==currentTic;
	}
}
class SDNS{
	constructor(params={}){
		// agents.push(this);
		this.VERSION='SDN-S1';
		var agent=this;
		this.params=Object.assign({
			crt:2,
			fs:0,
			lr:0.1	//if lr:2, model is frequency-based, not error-driven
		},params);
		if(this.params.lr===2)this.updateU=this.updateUfreq;
		else this.updateU=this.updateUerror;
		this.inputs={};
		this.currentTic=-1;
	}
	chunkActivate(chunk){
		if(chunk.lastActivationTime!=this.currentTic){ //ensure one activation per tic
			//activate chunk
			this.state.push(chunk);
			chunk._mostSpecific=true;
			chunk.lastActivationTime=this.currentTic;
			++chunk.f;
			//pass activation to subchunks
			for(var subchunk of chunk.subchunks){
				if(subchunk.allowActivation(this.currentTic)){
					subchunk.parent1._mostSpecific=false;
					subchunk.parent2._mostSpecific=false;
					this.chunkActivate(subchunk);
				}
			}
		}
	}
	learnChunking(){ //TODO: maybe have a param for number of learned chunks 
		//learn new chunks
		var chunk,chunkActivation,maxActivation1=-1,maxActivation2=-1,bestChunk1,bestChunk2;
		for(chunk of this.state){
			if(chunk._mostSpecific){
				//pick 2 most active chunks
				if(chunk.f>this.params.crt){
					chunkActivation=chunk.f+Math.random();
					if(chunkActivation>maxActivation1){
						bestChunk2=bestChunk1;
						maxActivation2=maxActivation1;
						bestChunk1=chunk;
						maxActivation1=chunkActivation;
					}else if(chunkActivation>maxActivation2){
						bestChunk2=chunk;
						maxActivation2=chunkActivation;
					}
				}
			}
		}
		//create new child from two most active parents
		if(bestChunk2!==undefined){
			new SChunk(bestChunk1,bestChunk2);
		}
	}
	updateUerror(chunk,output){
		var u=chunk.outputs[output]||0;
		chunk.outputs[output]=u+this.params.lr*(1-u);
	}
	updateUfreq(chunk,output){
		chunk.outputs[output]=(chunk.outputs[output]||0)+1;
	}
	learnOutput(output){
		for(var chunk of this.state)
			this.updateU(chunk,output);
	}
	perceive(perceptualCues){
		this.currentTic++;
		this.state=[];
		var chunk;
		for(var chunkName of perceptualCues){
			chunk=this.inputs[chunkName]||(this.inputs[chunkName]=new SChunk());
			this.chunkActivate(chunk);
		}
	}
	expose(inputs,epochs=1){
		for(var epoch=0;epoch<epochs;epoch++){
			if(epoch)shuffle(inputs);
			for(var input of inputs){
				this.perceive(input);
				this.learnChunking();
			}
		}
	}
	learnOutputs(cases,epochs=1){
		for(var epoch=0;epoch<epochs;epoch++){
			for(var io of cases){
				this.perceive(io[0]);
				this.learnOutput(io[1]);
			}
		}
	}
	decide(){
		var output,outputU=new Proxy({},{get:(o,k)=>o[k]||0});
		for(var chunk of this.state){
			if(chunk._mostSpecific||this.params.fs){
				for(output in chunk.outputs){
					outputU[output]+=chunk.outputs[output];
				}
			}
		}
		var bestU=-1,best;
		for(output in outputU){
			outputU[output]=outputU[output]+(Math.random()*.0000001);
			if(outputU[output]>bestU){
				best=output;
				bestU=outputU[output];
			}
		}
		return best;
	}
	train(cases,epochs=1,epochsO=1){
		this.expose(cases.map(io=>io[0]),epochs);
		this.learnOutputs(cases,epochsO);
	}
	predict(inputs){
		var predictions=[];
		for(var input of inputs){
			this.perceive(input);
			predictions.push(this.decide());
		}
		return predictions;
	}
}


if(typeof(window)==='undefined'){
	// /////////////////////////////////////////////////////////////
	// // exit handler
	// function exitHandler(options,err){
		// var agent;
		// while(agent=agents.pop())
			// agents[i].end();
		// if(err)console.error(err.stack);
		// if(options.exit)process.exit();
	// }
	// //starts handler on exit
	// process.on('exit', exitHandler.bind(null));
	// //catch ctrl+c event
	// process.on('SIGINT', exitHandler.bind(null,{exit:true}));
	// /////////////////////////////////////////////////////////////

	try{
		exports.CRRL=CRRL;
	}catch(e){}

	// var a=new CRRL(['left','right'],{},'try',1);

	// for(var i=0;i<1000;i++){
		// a.step(['a','b','c','d'],0)
		// a.step(['g'],1)
	// }

	// console.log('done');

	// a.end();
}
